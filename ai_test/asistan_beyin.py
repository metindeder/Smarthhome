import sys
import os
import ctypes
import time
from llama_cpp import Llama, LlamaGrammar

# =============================================================================
# 1. İŞLEMCİ ÖNCELİĞİ (Windows)
# =============================================================================
if sys.platform == "win32":
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
    )
    print("🔥 İşlemci önceliği HIGH_PRIORITY seviyesine çekildi.")

# =============================================================================
# 2. JSON GRAMMAR — llama.cpp GBNF (0.3.x uyumlu)
#    • Compact JSON: çıktıda ws yok → daha az token → daha hızlı
#    • Aksiyon 7 değerle kısıtlı — halüsinasyon imkânsız
#    • deger-val: number önce gelir, sonra string (22 → int, not "22")
#    NOT: llama_cpp 0.3.34'te LlamaRAMCache + grammar crash yapıyor,
#         yerel KV-cache ısınma çağrısı kullanıyoruz.
# =============================================================================
JSON_GRAMMAR = r"""
root        ::= "{\"cihaz\":" string ",\"aksiyon\":" aksiyon-val ",\"deger\":" deger-val "}"
aksiyon-val ::= "\"ac\"" | "\"kapat\"" | "\"artir\"" | "\"azalt\"" | "\"ayarla\"" | "\"durdur\"" | "\"baslat\""
deger-val   ::= "null" | [0-9]+ ("." [0-9]+)? | "\"" [^"]* "\""
string      ::= "\"" [^"]* "\""
"""
grammar = LlamaGrammar.from_string(JSON_GRAMMAR)
print("✅ JSON Grammar yüklendi.")

# =============================================================================
# 3. MODEL YÜKLEME — Qwen 2.5-0.5B Instruct Q2_K
#    Tüm modeller/quant'lar arasında en iyi sonuç:
#    ~971 ms ısınma | ~686 ms warm avg | ~29 tok/s | 5/5 doğruluk
#    Q4_K_M'den isınma ~330ms daha hızlı, inference hızı identik.
# =============================================================================
MODEL_YOLU = "../models/qwen2.5-0.5b-instruct-q2_k.gguf"

_cpu = os.cpu_count() or 4
N_THREADS = min(_cpu, 6)   # Benchmark: 6 thread bu sistemde en hızlı (8+ koordinasyon yükü yaratıyor)

print(f"\n🧠 Model yükleniyor: {MODEL_YOLU.split('/')[-1]}")
print(f"   CPU çekirdek: {_cpu}  →  N_THREADS: {N_THREADS}")

llm = Llama(
    model_path=MODEL_YOLU,
    n_ctx=512,
    n_batch=512,
    n_threads=N_THREADS,
    n_threads_batch=N_THREADS,
    n_gpu_layers=-1,
    use_mlock=True,
    chat_format="chatml",
    flash_attn=True,
    type_k=8,              # 🚀 KV-cache kuantizasyonu: bellek bant genişliği azalır
    type_v=8,
    verbose=False,
)

print("✅ Model hazır!")
print("-" * 55)

# =============================================================================
# 4. SİSTEM YÖNERGESİ — Qwen 2.5-1.5B için optimize
#    • Geçerli aksiyon değerleri listeleniyor
#    • Kritik eşleştirmeler (open→ac, açar mısın→ac) örnekle pekiştiriliyor
#    • Örnekler compact formatta → "No spaces" talimatıyla uyumlu
# =============================================================================
SYSTEM_PROMPT = (
    "You are a smart home voice command parser. "
    "Output ONLY a compact JSON object with exactly three keys:\n"
    "  \"cihaz\"  : device name in Turkish (e.g. \"TV\", \"lamba\", \"klima\", \"muzik\")\n"
    "  \"aksiyon\": one of: \"ac\", \"kapat\", \"artir\", \"azalt\", \"ayarla\", \"durdur\", \"baslat\"\n"
    "  \"deger\"  : number or null\n\n"
    "Mapping rules:\n"
    "  open/turn on/aç/açar mısın → \"ac\"\n"
    "  close/turn off/kapat       → \"kapat\"\n"
    "  increase/yükselt/artır     → \"artir\"\n"
    "  decrease/azalt/düşür       → \"azalt\"\n"
    "  set/ayarla/yap             → \"ayarla\"\n"
    "  stop/durdur                → \"durdur\"\n"
    "  start/başlat/oynat         → \"baslat\"\n\n"
    "Examples (compact, no spaces):\n"
    "  'open the tv'          → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'turn on the tv'       → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Televizyonu açar mısın?' → {\"cihaz\":\"TV\",\"aksiyon\":\"ac\",\"deger\":null}\n"
    "  'Mutfak ışıklarını kapat' → {\"cihaz\":\"mutfak ışıkları\",\"aksiyon\":\"kapat\",\"deger\":null}\n"
    "  'Klima sıcaklığını 22 yap' → {\"cihaz\":\"klima\",\"aksiyon\":\"ayarla\",\"deger\":22}\n"
    "  'Müziği durdur'        → {\"cihaz\":\"muzik\",\"aksiyon\":\"durdur\",\"deger\":null}\n\n"
    "No spaces in output JSON. Raw JSON only."
)

# =============================================================================
# 5. PROMPT KV-CACHE ISITMA
#    llama.cpp, aynı token prefix'ini tekrar işlemez.
#    Bu çağrı sistem promptunu KV-cache'e işler;
#    sonraki komutlar yalnızca kullanıcı mesajını (~10-20 tok) işler.
# =============================================================================
print("\n🔥 Sistem prompt KV-cache'e alınıyor (ısınma çağrısı)...")
_t_isinma = time.perf_counter()
llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "hazır"},
    ],
    max_tokens=1,
    temperature=0.0,
    grammar=grammar,
)
_isinma_ms = (time.perf_counter() - _t_isinma) * 1000
print(f"✅ KV-cache hazır! (ısınma süresi: {_isinma_ms:.0f} ms)")
print("-" * 55)

# =============================================================================
# 6. HIZLI REGEX PARSER
#    Tanınan cihaz + aksiyon çiftleri için LLM'i tamamen bypass eder.
#    Her ikisi de eşleşirse: <5 ms  |  Biri eksikse: LLM'e düş
# =============================================================================
import re as _re, json as _json

# Cihaz desenleri — uzun/spesifik olanlar önce gelir
_CIHAZLAR = [
    (r'mutfak\s+ışıklar',                'mutfak ışıkları'),
    (r'salon\s+ışıklar',                 'salon ışıkları'),
    (r'yatak\s+oda\S*\s+ışıklar',        'yatak odası ışıkları'),
    (r'tv|televizyon|television',         'TV'),
    (r'klima|air\s*cond',                 'klima'),
    (r'müz[iı]|music|speaker',           'muzik'),
    (r'perde|curtain|blind',             'perde'),
    (r'vantilatör|fan\b',                'fan'),
    (r'fırın|oven',                      'fırın'),
    (r'dishwasher|bulaşık\s+mak',        'bulaşık makinesi'),
    (r'washing\s+machine|çamaşır\s+mak', 'çamaşır makinesi'),
    (r'lamba|ışıklar?|ampul|light|lamp', 'lamba'),
]

# Aksiyon desenleri — önce spesifik/uzun olanlar, "aç" en sona
_AKSIYONLAR = [
    (r'kapat|kapa\b|close|turn\s+off|söndür',      'kapat'),
    (r'artır|y?ükselt|increase',                   'artir'),  # 🎯 Güncellenen satır
    (r'azalt|düşür|kıs\b|decrease|lower',        'azalt'),
    (r'ayarla|yap\b|set\b|değiştir|sabitle',       'ayarla'),
    (r'durdur|dur\b|stop|pause|beklet',            'durdur'),
    (r'başlat|çalıştır|oynat|play|start',         'baslat'),
    (r'aç|open|turn\s+on|yak\b',                   'ac'),     # en sona — diğerlerinden sonra kontrol
]

_SAYI_RE = _re.compile(r'\b(\d+(?:[.,]\d+)?)\b')

def regex_parse(komut: str) -> dict | None:
    """
    Hizli regex parser.
    Eşleşme bulursa dict döner, bulamazsa None → LLM'e düş.
    """
    k = komut.lower()

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

    if not cihaz or not aksiyon:   # ikisi de gerekli
        return None

    m = _SAYI_RE.search(k)
    deger = int(float(m.group(1).replace(',', '.'))) if m else None

    return {"cihaz": cihaz, "aksiyon": aksiyon, "deger": deger}


# =============================================================================
# 7. ÖĞRENİCİ CACHE — Self-Learning Sistemi
#    LLM yeni bir komut çözdükçe kalipları JSON dosyasına kaydeder.
#    Sonraki aynı komut: <0.1 ms (tam eşleşme) ya da <1 ms (regex).
# =============================================================================
_STOPWORDS = {
    # Türkçe
    "bir", "bu", "şu", "ve", "ile", "için", "de", "da", "ki", "mu", "mı",
    "mi", "mü", "misin", "mısın", "musun", "müsin", "bana", "beni", "lütfen",
    "acaba", "şu", "onu", "sesi", "biraz",
    # İngilizce
    "the", "a", "an", "you", "can", "could", "would", "please", "start",
    "turn", "make", "get", "put", "let", "set",
}

class OgreniciCache:
    """
    İki katmanlı öğrenme:
      1. Tam eşleşme: normalize edilmiş komut → anında sonuç (<0.1 ms)
      2. Cihaz deseni: yeni cihaz kelimesini _CIHAZLAR listesine dinamik ekler
    """
    DOSYA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ogrenilen_komutlar.json")

    def __init__(self):
        self._veri = self._yukle()
        self._tam: dict  = self._veri["tam_eslesmeler"]
        self._cihazlar: list = self._veri["cihaz_desenleri"]
        # Ogrenilenler calisan regex listesine ekle
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
        return _re.sub(r'[?!.,;:\-]', '', metin.lower().strip())

    def ara(self, komut: str) -> dict | None:
        """Tam eşleşme cache'inde arar. Bulursa dict, yoksa None."""
        return self._tam.get(self._normalize(komut))

    def ogren(self, komut: str, sonuc: dict):
        """
        LLM sonuçundan yeni kalıp öğrenir:
        - Tam eşleşme olarak kaydeder
        - Cihazı statik listede yoksa yeni desen çıkarır
        """
        norm = self._normalize(komut)
        cihaz = sonuc.get("cihaz", "")

        # 1. Tam eşleşme kaydet
        if norm not in self._tam:
            self._tam[norm] = sonuc

        # 2. Cihaz zaten bilinen bi desende var mı?
        cihaz_biliniyor = any(
            _re.search(d[0], cihaz.lower(), _re.IGNORECASE) for d in _CIHAZLAR
        )
        if not cihaz_biliniyor and cihaz:
            # Komuttaki anlamlı kelimeleri çıkar
            kelimeler = [
                k for k in self._normalize(komut).split()
                if len(k) > 2 and k not in _STOPWORDS
            ]
            if kelimeler:
                # Keli melerin ilk 2'sini desen olarak kullan
                desen = r'\s+'.join(
                    _re.escape(k[:6]) for k in kelimeler[:2]
                )
                yeni = {"desen": desen, "isim": cihaz}
                self._cihazlar.append(yeni)
                _CIHAZLAR.append((desen, cihaz))   # çalışan listeye de ekle
                print(f"  📚 Öğrenildi → desen: '{desen}' â cihaz: '{cihaz}'")

        self._kaydet()

    def istatistik(self) -> str:
        return (f"{len(self._tam)} tam eşleşme | "
                f"{len(self._cihazlar)} öğrenilmiş cihaz deseni")


cache = OgreniciCache()
print(f"💡 Öğrenen cache yüklendi: {cache.istatistik()}")
print("-" * 55)

# =============================================================================
# 8. ÇIKARIM FONKSİYONU — Hybrid: Cache → Regex → LLM → Öğren
# =============================================================================
def komut_isle(komut_metni: str, etiket: str = "TEST"):
    print(f"\n[{etiket}]")
    print(f"  🎤  Komut : '{komut_metni}'")

    t0 = time.perf_counter()

    # ——— YOL 1: Öğrenilmiş tam eşleşme cache (<0.1 ms) ———
    ogrenilmis = cache.ara(komut_metni)
    if ogrenilmis:
        sure_ms = (time.perf_counter() - t0) * 1000
        print(f"  ⏱️  Süre       : {sure_ms:.2f} ms  [📂 CACHE]")
        print(f"  📦  JSON       : {_json.dumps(ogrenilmis, ensure_ascii=False)}")
        print("-" * 55)
        return ogrenilmis

    # ——— YOL 2: Regex (statik + öğrenilmiş desenler, <1 ms) ———
    hizli = regex_parse(komut_metni)
    if hizli:
        sure_ms = (time.perf_counter() - t0) * 1000
        print(f"  ⏱️  Süre       : {sure_ms:.2f} ms  [⚡ REGEX]")
        print(f"  📦  JSON       : {_json.dumps(hizli, ensure_ascii=False)}")
        print("-" * 55)
        return hizli

    # ——— YOL 3: LLM (~700 ms, ilk kez görülen komut) ———
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

    print(f"  ⏱️  Süre       : {sure_ms:.0f} ms  [🧠 LLM]")
    print(f"  ⚡  Throughput : {tok_per_sn:.1f} tok/s  ({completion_tok} gen + {prefill_tok} prefill)")
    print(f"  📦  JSON       : {json_ciktisi}")
    print("-" * 55)

    # ——— YOL 3b: LLM sonuçtan öğren ———
    try:
        sonuc_dict = _json.loads(json_ciktisi)
        cache.ogren(komut_metni, sonuc_dict)
    except _json.JSONDecodeError:
        pass  # Geçersiz JSON — öğrenme

    return json_ciktisi

# =============================================================================
# 9. TEST SENARYOLARI
# =============================================================================
# ⚡ Regex yolu
komut_isle("open the tv",                   "TEST 1 — REGEX")
komut_isle("Mutfak ışıklarını kapat",       "TEST 2 — REGEX")
komut_isle("Televizyonu açar mısın?",       "TEST 3 — REGEX")
komut_isle("Klima sıcaklığını 22 yap",      "TEST 4 — REGEX")
komut_isle("Müziği durdur",                 "TEST 5 — REGEX")
# 🧠 LLM yolu — yeni komut, cache'e eklenecek
komut_isle("Çay makinesini çalıştır",          "TEST 6 — LLM (öğreniyor)")
# 📂 Cache yolu — ayni komut tekrar geldi
komut_isle("Çay makinesini çalıştır",          "TEST 7 — CACHE (ilk seferden sonra)")