"""
app/core/asistan_beyin.py — Akilli Ev AI Komut Isleme Motoru

Hybrid inference pipeline:
  Yol 1: Ogrenilmis tam eslesme cache (<0.1 ms)
  Yol 2: Statik + ogrenilenler regex (<1 ms)
  Yol 3: LLM — Qwen2.5-0.5B Instruct Q2_K (~700 ms, ilk kez gorulun komut)

Kullanim:
    from app.core.asistan_beyin import komut_isle
    sonuc = komut_isle("Lambay ac")
"""

import sys
import os
import ctypes
import time
import re as _re
import json as _json
import paho.mqtt.publish as mqtt_publish
from llama_cpp import Llama, LlamaGrammar
import paho.mqtt.client as mqtt_client
import threading


# =============================================================================
# 1. ISLEMCI ONCELIGI (Windows)
# =============================================================================
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
    )
    print("Islemci onceligi HIGH_PRIORITY seviyesine cekildi.")

# =============================================================================
# 2. JSON GRAMMAR
# =============================================================================
# Cihazlari sadece bizim belirlediklerimizle sinirlandiriyoruz. LLM baska kelime uretemez.
JSON_GRAMMAR = r"""
root        ::= "{\"cihaz\":" cihaz-val ",\"aksiyon\":" aksiyon-val ",\"deger\":" deger-val "}"
cihaz-val   ::= "\"TV\"" | "\"klima\"" | "\"lamba\"" | "\"muzik\"" | "\"mutfak isiklari\"" | "\"yatak odasi isiklari\"" | "\"salon isiklari\"" | "\"perde\"" | "\"firin\"" | "\"fan\"" | "\"bilinmiyor\""
aksiyon-val ::= "\"ac\"" | "\"kapat\"" | "\"artir\"" | "\"azalt\"" | "\"ayarla\"" | "\"durdur\"" | "\"baslat\"" | "\"bilinmiyor\""
deger-val   ::= "null" | [0-9]+ ("." [0-9]+)?
"""
grammar = LlamaGrammar.from_string(JSON_GRAMMAR)
print("sıkı JSON Grammar yuklendi.")

# =============================================================================
# 3. MODEL YÜKLEME
#    Model dosyasi: Smarthome/models/qwen2.5-0.5b-instruct-q2_k.gguf
# =============================================================================
_DOSYA_DIR = os.path.dirname(os.path.abspath(__file__))          # app/core/
_PROJE_KOK = os.path.dirname(os.path.dirname(_DOSYA_DIR))        # Smarthome/
MODEL_YOLU = os.path.join(_PROJE_KOK, "models", "qwen2.5-0.5b-instruct-q2_k.gguf")

_cpu = os.cpu_count() or 4
N_THREADS = min(_cpu, 6)

print(f"\nModel yukleniyor: {os.path.basename(MODEL_YOLU)}")
print(f"   CPU cekirdek: {_cpu}  ->  N_THREADS: {N_THREADS}")

llm = Llama(
    model_path=MODEL_YOLU,
    n_ctx=1024,
    n_batch=512,
    n_threads=N_THREADS,
    n_threads_batch=N_THREADS,
    n_gpu_layers=-1,
    use_mlock=True,
    chat_format="chatml",
    flash_attn=True,
    type_k=8,
    type_v=8,
    verbose=False,
)

print("Model hazir!")
print("-" * 55)

# =============================================================================
# 4. SISTEM YONERGESI
# =============================================================================
SYSTEM_PROMPT = (
    "You are a strict smart home voice command parser. "
    "Voice transcription may contain typos or phonetic errors (e.g., 'Kılıma' -> 'klima', 'filmi' -> 'TV', 'Işık' -> 'lamba'). "
    "Fix these typos and map them to the nearest VALID DEVICE. "
    "CRITICAL RULE: If the input is conversational, a question, random noise, or NOT a smart home command (e.g., 'nasılsın', 'ne yapıyorsun', 'iyi misin'), you MUST output 'bilinmiyor' for both cihaz and aksiyon. "
    "Output ONLY a compact JSON object with exactly three keys: \"cihaz\", \"aksiyon\", \"deger\".\n\n"
    "VALID DEVICES (must be one of these):\n"
    "  TV, klima, lamba, muzik, mutfak isiklari, yatak odasi isiklari, salon isiklari, perde, firin, fan, bilinmiyor\n\n"
    "Mapping rules:\n"
    "  open/turn on/ac/acar misin -> \"ac\"\n"
    "  close/turn off/kapat       -> \"kapat\"\n"
    "  increase/yukselt/artir     -> \"artir\"\n"
    "  decrease/azalt/dusur       -> \"azalt\"\n"
    "  set/ayarla/yap             -> \"ayarla\"\n"
    "  stop/durdur                -> \"durdur\"\n"
    "  start/baslat/oynat         -> \"baslat\"\n\n"
    "Examples (compact, no spaces):\n"
    "  'open the tv'              -> {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Televizyonu acar misin?'  -> {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Mutfak isiklar kapat'     -> {\"cihaz\":\"mutfak isiklari\",\"aksiyon\":\"kapat\",\"deger\":null}\n"
    "  'Klima sicakligi 22 yap'   -> {\"cihaz\":\"klima\",\"aksiyon\":\"ayarla\",\"deger\":22}\n"
    "  'Muzigi durdur'            -> {\"cihaz\":\"muzik\",\"aksiyon\":\"durdur\",\"deger\":null}\n"
    "  'Kılıma'ya aç'             -> {\"cihaz\":\"klima\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'filmi aç'                 -> {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Bir daha iyi misin?'      -> {\"cihaz\":\"bilinmiyor\",\"aksiyon\":\"bilinmiyor\",\"deger\":null}\n\n"
    "No spaces in output JSON. Raw JSON only."
)

# =============================================================================
# 5. KV-CACHE ISITMA
# =============================================================================
print("\nSistem prompt KV-cache'e aliniyor (isinma cagrisi)...")
_t_isinma = time.perf_counter()
llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "hazir"},
    ],
    max_tokens=1,
    temperature=0.0,
    grammar=grammar,
)
_isinma_ms = (time.perf_counter() - _t_isinma) * 1000
print(f"KV-cache hazir! (isinma suresi: {_isinma_ms:.0f} ms)")
print("-" * 55)

# =============================================================================
# 6. HIZLI REGEX PARSER
# =============================================================================
_CIHAZLAR = [
    (r'mutfak\s+[iı]s[iı]klar',                          'mutfak isiklari'),
    (r'salon\s+[iı]s[iı]klar',                           'salon isiklari'),
    (r'yatak\s+oda\S*\s+[iı]s[iı]klar',                 'yatak odasi isiklari'),
    (r'tv|televizyon|television',                         'TV'),
    (r'klima|air\s*cond',                                 'klima'),
    (r'mü?z[iı]k?|music|speaker',                        'muzik'),
    (r'perde|curtain|blind',                              'perde'),
    (r'vantilatör|vantilator|fan\b',                     'fan'),
    (r'f[iı]r[iı]n|oven',                                'firin'),
    (r'dishwasher|bula[sş][iı]k\s+mak',                 'bulasik makinesi'),
    (r'washing\s+machine|çama[sş][iı]r\s+mak|camasir\s+mak', 'camasir makinesi'),
    (r'lamba|[iı]s[iı]klar?|ampul|light|lamp',           'lamba'),
    (r'robot|[sş]ü?pürge|supurge',                       'robot supurge'),
    (r'kap[iı]|kilit',                                   'kapi kilidi'),
    (r'kahve|espresso',                                   'kahve makinesi'),
]

_AKSIYONLAR = [
    (r'kapat|kapa\b|close|turn\s+off|söndür|sondur',     'kapat'),
    (r'art[iı]r|yükselt|ukselt|increase',                'artir'),
    (r'azalt|düşür|dusur|k[iı][sş]\b|decrease|lower',   'azalt'),
    (r'ayarla|yap\b|set\b|de[gğ]i[sş]tir|sabitle',      'ayarla'),
    (r'durdur|dur\b|stop|pause|beklet',                  'durdur'),
    (r'ba[sş]lat|çal[iı][sş]t[iı]r|oynat|play|start',   'baslat'),
    (r'aç|ac\b|open|turn\s+on|yak\b',                   'ac'),
    (r'kilitle',                                          'kilitle'),
    (r'temizle|süpür|supur',                              'temizle'),
]

_SAYI_RE = _re.compile(r'\b(\d+(?:[.,]\d+)?)\b')


def regex_parse(komut: str) -> dict | None:
    k = komut.lower()  # Turkce karakterler korunur, sadece kucuk harf

    cihaz = None
    for desen, isim in _CIHAZLAR:
        if _re.search(desen, k, _re.IGNORECASE):
            cihaz = isim
            break

    aksiyon = None
    for desen, eylem in _AKSIYONLAR:
        if _re.search(desen, k, _re.IGNORECASE):
            aksiyon = eylem
            break

    if not cihaz or not aksiyon:
        return None

    m = _SAYI_RE.search(k)
    deger = int(float(m.group(1).replace(',', '.'))) if m else None

    return {"cihaz": cihaz, "aksiyon": aksiyon, "deger": deger}


# =============================================================================
# 6.5 GÜVENİLİRLİK KONTROL KATMANI
# =============================================================================
# LLM'e gitmeden önce ve sonra iki kapı:
#   KAPI-1 (pre-LLM)  : Giriş metni akıllı ev komutu gibi görünüyor mu?
#   KAPI-2 (post-LLM) : LLM çıktısındaki cihaz/aksiyon girişle örtüşüyor mu?

# Akıllı ev ile ilgili anahtar kelimeler (skor için)
_AE_ANAHTAR = [
    r'tv|televizyon|ekran|filmi?|diziy?i?',
    r'klima|[iı]s[iı]tma|so[gğ]utma|hava',
    r'lamba|[iı][sş][iı]k|ampul|ayd[iı]nlat',
    r'mü?z[iı]k?|[sş]ark[iı]|radyo|ses|çal|oynat|durdur|duraklat',
    r'perde|jaluz[iı]',
    r'fan|vantilat[oö]r',
    r'f[iı]r[iı]n|[fF]irin',
    r'mutfak|salon|yatak\s*oda',
    r'ac|aç|kapat|kapa\b|yak\b|söndür|sondur',
    r'art[iı]r|azalt|yükselt|dü[sş]ür|ayarla',
    r'ba[sş]lat|durdur|dur\b|beklet',
    r'ev|oda|[sS]mart|home|cihaz',
]
_AE_ANAHTAR_RE = [_re.compile(p, _re.IGNORECASE) for p in _AE_ANAHTAR]

# Minimum eşik: kaç farklı kategoride eşleşme gerekir?
_GUVEN_ESIGI = 1  # En az 1 kategori eşleşmeli (çok düşük eşik, çok esnek)


def _giris_guven_skoru(metin: str) -> int:
    """Giriş metninin akıllı ev komutu kategorileriyle kaç farklı kategoride örtüştüğünü döndürür."""
    return sum(1 for r in _AE_ANAHTAR_RE if r.search(metin))


# Cihaz adı -> giriş metni için kontrol kelimeleri eşlemesi
# NOT: Whisper bazen kelimeleri ortadan böler (örn: 'klimayı' -> 'kli mayi').
#      Bu yüzden her cihaz için fonetik parçalı kalıplar da eklendi.
_CIHAZ_DOGRULAMA = {
    'TV':                    [r'tv|telev[iı]\s*z|televizyon|ekran|filmi?|diziy?i?'],
    'klima':                 [r'klima|kl[iı]\s*m|[iı]s[iı]tma|so[gğ]utma|hava'],
    'lamba':                 [r'lamba|lam\s*ba|[iı][sş][iı]k|ampul|ayd[iı]nlat|yak\b|söndür'],
    'muzik':                 [r'mü?\s*z[iı]\s*k?|[sş]ark[iı]|radyo|ses|çal|oynat|durdur|duraklat'],
    'mutfak isiklari':       [r'mutfak|mut\s*fak|[iı][sş][iı]k|lamba'],
    'salon isiklari':        [r'salon|sal\s*on|[iı][sş][iı]k|lamba'],
    'yatak odasi isiklari':  [r'yatak|ya\s*tak|oda|[iı][sş][iı]k|lamba'],
    'perde':                 [r'perde|per\s*de|jaluz[iı]'],
    'firin':                 [r'f[iı]\s*r[iı]\s*n'],
    'fan':                   [r'fan\b|vantilat[oö]r'],
}

# Eylem kalıpları — giriş eylem bağlamını doğrulamak için
_AKSIYON_DOGRULAMA = {
    'ac':      r'aç|ac\b|open|turn\s*on|yak\b',
    'kapat':   r'kapat|kapa\b|close|turn\s*off|söndür|sondur',
    'artir':   r'art[iı]r|yükselt|increase',
    'azalt':   r'azalt|dü[sş]ür|decrease',
    'ayarla':  r'ayarla|set\b',
    'durdur':  r'durdur|dur\b|stop|pause',
    'baslat':  r'ba[sş]lat|oynat|play|start|çal',
}


def _cikti_dogrula(komut: str, sonuc: dict) -> bool:
    """LLM çıktısının giriş metniyle mantıksal örtüşüp örtüşmediğini kontrol eder.

    True  -> çıktı güvenilir, komut işlenebilir
    False -> çıktı şüpheli, komut reddedilmeli

    Strateji:
      - Aksiyon giriş metninde net olarak bulunuyorsa (örn: 'aç', 'kapat'),
        Whisper'ın fonetik hatası olduğu varsayılır ve cihaz doğrulaması atlanır.
        Çünkü net eylem + bulanık cihaz adı = fonetik gürültü, sahte pozitif değil.
      - Hem cihaz hem aksiyon için girişte hiçbir ipucu yoksa -> red.
    """
    cihaz = sonuc.get("cihaz", "bilinmiyor")
    aksiyon = sonuc.get("aksiyon", "bilinmiyor")

    # bilinmiyor zaten güvenli (reddediliyor üst katta)
    if cihaz == "bilinmiyor" or aksiyon == "bilinmiyor":
        return True

    # Aksiyon girişte net geçiyor mu? (örn: 'aç', 'kapat', 'durdur')
    aksiyon_deseni = _AKSIYON_DOGRULAMA.get(aksiyon)
    aksiyon_eslesti = (
        bool(_re.search(aksiyon_deseni, komut, _re.IGNORECASE))
        if aksiyon_deseni else False
    )

    if aksiyon_eslesti:
        # Net aksiyon var → fonetik cihaz hatası kabul edilir, geç
        return True

    # Aksiyon da belirsizse cihaz için ek kontrol yap
    desenler = _CIHAZ_DOGRULAMA.get(cihaz, [])
    if desenler:
        eslesti = any(_re.search(p, komut, _re.IGNORECASE) for p in desenler)
        if not eslesti:
            return False  # Ne cihaz ne aksiyon girişte geçiyor -> şüpheli

    return True


# =============================================================================
# 7. OGRENICI CACHE
# =============================================================================
_STOPWORDS = {
    "bir", "bu", "şu", "ve", "ile", "için", "de", "da", "ki", "mü", "mi",
    "misin", "misiniz", "musun", "bana", "beni", "lütfen", "acaba", "onu",
    "the", "a", "an", "you", "can", "could", "would", "please",
    "turn", "make", "get", "put", "let", "set",
}


class OgreniciCache:
    """
    Iki katmanli ogrenme:
      1. Tam eslesme: normalize edilmis komut -> aninda sonuc (<0.1 ms)
      2. Cihaz deseni: yeni cihaz kelimesini _CIHAZLAR listesine dinamik ekler
    """
    DOSYA = os.path.join(_DOSYA_DIR, "ogrenilen_komutlar.json")

    def __init__(self):
        self._veri = self._yukle()
        self._tam: dict  = self._veri["tam_eslesmeler"]
        self._cihazlar: list = self._veri["cihaz_desenleri"]
        for g in self._cihazlar:
            if not any(d[0] == g["desen"] for d in _CIHAZLAR):
                _CIHAZLAR.append((g["desen"], g["isim"]))

    def _yukle(self) -> dict:
        try:
            with open(self.DOSYA, encoding="utf-8") as f:
                return _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return {"tam_eslesmeler": {}, "cihaz_desenleri": []}

    def _kaydet(self):
        with open(self.DOSYA, "w", encoding="utf-8") as f:
            _json.dump(self._veri, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize(metin: str) -> str:
        # Noktalama temizle, kucuk harf yap; Turkce karakterler korunur
        temiz = _re.sub(r'[?!.,;:\-]', '', metin.strip())
        return temiz.lower()

    def ara(self, komut: str) -> dict | None:
        return self._tam.get(self._normalize(komut))

    def ogren(self, komut: str, sonuc: dict):
        norm = self._normalize(komut)
        cihaz = sonuc.get("cihaz", "")

        if norm not in self._tam:
            self._tam[norm] = sonuc

        cihaz_biliniyor = any(
            _re.search(d[0], cihaz.lower(), _re.IGNORECASE) for d in _CIHAZLAR
        )
        if not cihaz_biliniyor and cihaz:
            kelimeler = [
                k for k in self._normalize(komut).split()
                if len(k) > 2 and k not in _STOPWORDS
            ]
            if kelimeler:
                desen = r'\s+'.join(_re.escape(k[:6]) for k in kelimeler[:2])
                yeni = {"desen": desen, "isim": cihaz}
                self._cihazlar.append(yeni)
                _CIHAZLAR.append((desen, cihaz))
                print(f"  Ogrenildi -> desen: '{desen}' => cihaz: '{cihaz}'")

        self._kaydet()

    def istatistik(self) -> str:
        return (
            f"{len(self._tam)} tam eslesme | "
            f"{len(self._cihazlar)} ogrenilmis cihaz deseni"
        )


cache = OgreniciCache()
print(f"Ogrenici cache yuklendi: {cache.istatistik()}")
print("-" * 55)




# =============================================================================
# 7.5 MQTT YAYINLAYICI (PUBLISHER)
# =============================================================================
MQTT_BROKER = "127.0.0.1" # Kendi bilgisayarında çalıştığı için localhost
MQTT_PORT = 1883

def mqtt_ilet(sonuc: dict):
    """Üretilen JSON komutunu ilgili topic'e fırlatır."""
    if not isinstance(sonuc, dict) or "cihaz" not in sonuc:
        return
        
    # Topic oluştururken cihaz adındaki boşlukları alt çizgi yapalım (örn: mutfak isiklari -> mutfak_isiklari)
    cihaz_adi = sonuc["cihaz"].replace(" ", "_")
    topic = f"ev/{cihaz_adi}/komut"
    payload = _json.dumps(sonuc, ensure_ascii=False)
    
    try:
        # Tek atımlık hızlı fırlatma
        mqtt_publish.single(topic, payload=payload, hostname=MQTT_BROKER, port=MQTT_PORT)
        print(f"  [MQTT OK] -> Topic: {topic} | Veri: {payload}")
    except Exception as e:
        print(f"  [MQTT HATA] Broker'a ulaşılamadı. Mosquitto açık mı? Detay: {e}")
# Teyit bekleyen komutları tutacağımız sözlük (Bekleme Odası)
bekleyen_komutlar = {}
bekleyen_lock = threading.Lock()

def on_durum_connect(client, userdata, flags, rc):
    client.subscribe("ev/+/durum") # Tüm cihazların durum kanalını dinle

from app.services.sesli_yanit import sesli_yanit  # Servisin import edildiğinden emin ol

def on_durum_message(client, userdata, msg):
    try:
        payload = _json.loads(msg.payload.decode('utf-8'))
        if payload.get("durum") == "basarili":
            cihaz = payload.get("cihaz")
            aksiyon = payload.get("aksiyon") # Aksiyon bilgisini de alıyoruz
            t_bitis = time.perf_counter()
            
            with bekleyen_lock:
                if cihaz in bekleyen_komutlar:
                    veri = bekleyen_komutlar.pop(cihaz)
                    komut = veri["komut"]
                    sonuc = veri["sonuc"]
                    stt_ms = veri["stt_ms"]
                    t0 = veri["t0"]
                    ogrenilecek = veri["ogrenilecek"]
                    
                    if ogrenilecek:
                        cache.ogren(komut, sonuc)
                    
                    # --- TOPLAM SÜRE HESABI (TTS BURAYA DAHİL EDİLMEZ) ---
                    asistan_ve_mqtt_ms = (t_bitis - t0) * 1000
                    toplam_ms = stt_ms + asistan_ve_mqtt_ms
                    
                    print(f"  [TEYİT] Cihaz onayladı -> Toplam Süre: {toplam_ms:.0f} ms")
                    print("-" * 55)
                    
                    # --- SÜRE HESAPLANDIKTAN SONRA SESLİ YANIT TETİKLENİR ---
                    sesli_yanit.basarili(cihaz=cihaz, aksiyon=aksiyon)
    except Exception:
        pass

# Asistanın dinleyici istemcisi (Arka planda çalışacak)
asistan_listener = mqtt_client.Client(client_id="asistan_beyin_dinleyici")
asistan_listener.on_connect = on_durum_connect
asistan_listener.on_message = on_durum_message
try:
    asistan_listener.connect(MQTT_BROKER, MQTT_PORT, 60)
    asistan_listener.loop_start() # Thread olarak arka planda dinlemeyi başlat
except Exception as e:
    print(f"  [MQTT DİNLEYİCİ HATA] Başlatılamadı: {e}")
# =============================================================================
# 8. CIKARI FONKSIYONU — Hybrid: Cache -> Regex -> LLM -> Ogren
# =============================================================================
import threading
import time
import json as _json

# Bekleyen komutlar ve kilit mekanizması
bekleyen_komutlar = {}
bekleyen_lock = threading.Lock()

def komut_isle(komut_metni: str, etiket: str = "KOMUT", stt_ms: float = 0.0) -> dict | str:
    print(f"\n[{etiket}]")
    print(f"  Komut : '{komut_metni}'")

    t0 = time.perf_counter()

   # YOL 1: Ogrenilmis tam eslesme cache
    ogrenilmis = cache.ara(komut_metni)
    if ogrenilmis:
        sure_ms = (time.perf_counter() - t0) * 1000
        print(f"  Sure  : {sure_ms:.2f} ms  [CACHE]")
        print(f"  JSON  : {_json.dumps(ogrenilmis, ensure_ascii=False)}")
        
        # --- BEKLEME ODASINA EKLE (Teyit için) ---
        cihaz = ogrenilmis.get("cihaz")
        with bekleyen_lock:
            bekleyen_komutlar[cihaz] = {
                "komut": komut_metni,
                "sonuc": ogrenilmis,
                "stt_ms": stt_ms,
                "t0": t0,
                "ogrenilecek": False # Zaten öğrenilmiş
            }
        
        mqtt_ilet(ogrenilmis)
        return ogrenilmis

  # YOL 2: Regex
    hizli = regex_parse(komut_metni)
    if hizli:
        sure_ms = (time.perf_counter() - t0) * 1000
        print(f"  Sure  : {sure_ms:.2f} ms  [REGEX]")
        print(f"  JSON  : {_json.dumps(hizli, ensure_ascii=False)}")
        
        cihaz = hizli.get("cihaz")
        with bekleyen_lock:
            bekleyen_komutlar[cihaz] = {
                "komut": komut_metni,
                "sonuc": hizli,
                "stt_ms": stt_ms,
                "t0": t0,
                "ogrenilecek": False # Regex zaten biliniyor, öğrenmeye gerek yok
            }
            
        mqtt_ilet(hizli)
        return hizli

    # YOL 3: LLM
    # --- KAPI-1: Giriş güven skoru (pre-LLM filtresi) ---
    skor = _giris_guven_skoru(komut_metni)
    if skor < _GUVEN_ESIGI:
        print(f"  [GÜVEN-KAPI-1] Skor {skor}/{len(_AE_ANAHTAR_RE)} — Akıllı ev komutu değil, LLM atlandı.")
        return {"cihaz": "bilinmiyor", "aksiyon": "bilinmiyor", "deger": None}

    sonuc = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": komut_metni},
        ],
        max_tokens=40,
        temperature=0.0,
        repeat_penalty=1.0,
        grammar=grammar,
    )

    sure_ms = (time.perf_counter() - t0) * 1000
    json_ciktisi   = sonuc["choices"][0]["message"]["content"].strip()
    completion_tok = sonuc["usage"]["completion_tokens"]
    prefill_tok    = sonuc["usage"]["prompt_tokens"]
    tok_per_sn     = (completion_tok / sure_ms) * 1000 if sure_ms > 0 else 0

    print(f"  Sure  : {sure_ms:.0f} ms  [LLM]")
    print(f"  Hiz   : {tok_per_sn:.1f} tok/s  ({completion_tok} gen + {prefill_tok} prefill)")
    print(f"  JSON  : {json_ciktisi}")

    try:
        sonuc_dict = _json.loads(json_ciktisi)

        # --- KAPI-2: Çıktı mantık doğrulama (post-LLM filtresi) ---
        if not _cikti_dogrula(komut_metni, sonuc_dict):
            print(f"  [GÜVEN-KAPI-2] Çıktı giriş metniyle örtüşmüyor — Komut reddedildi.")
            print(f"                 Cihaz '{sonuc_dict.get('cihaz')}' için girişte eşleşen kelime bulunamadı.")
            return {"cihaz": "bilinmiyor", "aksiyon": "bilinmiyor", "deger": None}

        if sonuc_dict.get("cihaz") != "bilinmiyor":
            cihaz = sonuc_dict.get("cihaz")
            # Cihazdan onay gelene kadar bekleme odasına alıyoruz
            with bekleyen_lock:
                bekleyen_komutlar[cihaz] = {
                    "komut": komut_metni,
                    "sonuc": sonuc_dict,
                    "stt_ms": stt_ms,
                    "t0": t0,
                    "ogrenilecek": True
                }
        mqtt_ilet(sonuc_dict)
        return sonuc_dict
    except _json.JSONDecodeError:
        return json_ciktisi



    
