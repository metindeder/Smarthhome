import streamlit as st
import requests
import datetime
import sys
import json
from pathlib import Path

# Proje kök dizinini (Smarthome/) yola ekle, böylece app.x modüllerini import edebiliriz
_PROJE_KOK = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJE_KOK))

# AI Beynini doğrudan içeri alıyoruz (Dışarıdan API çağrısı yapmadan yerel hızı kullanmak için)
from app.services.asistan_beyin import komut_isle

# =============================================================================
# 1. SAYFA AYARLARI VE CSS
# =============================================================================
st.set_page_config(page_title="Edge AI Akıllı Ev", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .terminal-box {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 5px;
        height: 300px;
        overflow-y: auto;
        border: 1px solid #333;
    }
    .device-card {
        background-color: #2b2b40;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #444;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-on { color: #00ff00; font-weight: bold; }
    .status-off { color: #ff4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SESSION STATE (DURUM YÖNETİMİ)
# =============================================================================
if "cihazlar" not in st.session_state:
    st.session_state.cihazlar = {
        "TV": {"durum": "Kapalı", "deger": None, "ikon": "📺"},
        "lamba": {"durum": "Kapalı", "deger": None, "ikon": "💡"},
        "klima": {"durum": "Kapalı", "deger": 22, "ikon": "❄️"},
        "muzik": {"durum": "Kapalı", "deger": None, "ikon": "🎵"},
        "perde": {"durum": "Kapalı", "deger": None, "ikon": "🪟"},
    }

if "terminal_loglari" not in st.session_state:
    st.session_state.terminal_loglari = [
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sistem başlatıldı...",
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Uç Yapay Zeka (Qwen) hazır.",
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] MQTT Bağlantısı (NanoMQ) bekleniyor..."
    ]

def log_ekle(mesaj: str):
    zaman = datetime.datetime.now().strftime('%H:%M:%S')
    st.session_state.terminal_loglari.insert(0, f"[{zaman}] {mesaj}")

# =============================================================================
# 3. YARDIMCI FONKSİYONLAR
# =============================================================================
def arayuz_guncelle(sonuc: dict):
    """AI'dan dönen sonuca göre arayüzdeki cihaz durumlarını günceller."""
    cihaz = sonuc.get("cihaz")
    aksiyon = sonuc.get("aksiyon")
    deger = sonuc.get("deger")

    if cihaz in st.session_state.cihazlar:
        if aksiyon in ["ac", "baslat"]:
            st.session_state.cihazlar[cihaz]["durum"] = "Açık"
        elif aksiyon in ["kapat", "durdur"]:
            st.session_state.cihazlar[cihaz]["durum"] = "Kapalı"
        
        if deger is not None:
            st.session_state.cihazlar[cihaz]["deger"] = deger

def komut_calistir(metin: str):
    """Metin komutunu alır, yapay zekaya işletir ve FastAPI'ye postalar."""
    if not metin.strip():
        return
        
    log_ekle(f"🎙️ Gelen Komut: '{metin}'")
    
    # 1. AI Beyni ile işle (Regex + Cache + LLM)
    t_start = datetime.datetime.now()
    sonuc_json = komut_isle(metin, etiket="DASHBOARD")
    sure_ms = (datetime.datetime.now() - t_start).total_seconds() * 1000
    
    # 2. Sonucu logla
    log_ekle(f"🧠 Çözümlendi ({sure_ms:.0f}ms): {json.dumps(sonuc_json, ensure_ascii=False)}")
    
    if sonuc_json.get("cihaz") != "bilinmiyor":
        # 3. FastAPI & MQTT'ye ilet
        try:
            res = requests.post("http://localhost:8000/api/v1/ai-komut", json=sonuc_json)
            if res.status_code == 200:
                log_ekle("📡 MQTT'ye iletildi: ev/komutlar/ai")
            else:
                log_ekle(f"⚠️ FastAPI Hatası: {res.status_code}")
        except requests.exceptions.ConnectionError:
            log_ekle("❌ FastAPI sunucusuna ulaşılamadı! Uvicorn çalışıyor mu?")

        # 4. Arayüz kartlarını güncelle
        arayuz_guncelle(sonuc_json)
    else:
        log_ekle("❌ Anlaşılamayan komut.")

# =============================================================================
# 4. UI TASARIMI
# =============================================================================
st.title("🏠 Edge AI Akıllı Ev Merkezi")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎙️ Sesli/Yazılı Komut Simülatörü")
    st.info("💡 Asistan_dongu.py arka planda mikrofonu dinlerken, buradan manuel komut da gönderebilirsiniz.")
    
    with st.form("komut_formu", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            kullanici_komutu = st.text_input("Asistana bir şey söyleyin...", placeholder="Örn: Salondaki klimayı 22 derece yap")
        with col_btn:
            submit_btn = st.form_submit_button("🚀 Gönder")
            
        if submit_btn and kullanici_komutu:
            komut_calistir(kullanici_komutu)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📟 Sistem Terminali (Loglar)")
    
    # Terminal Kutusu
    log_html = "<br>".join(st.session_state.terminal_loglari)
    st.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📱 Cihaz Durumları")
    
    # Cihazları grid şeklinde listele
    c1, c2 = st.columns(2)
    sutunlar = [c1, c2, c1, c2, c1]
    
    for i, (cihaz_adi, veri) in enumerate(st.session_state.cihazlar.items()):
        with sutunlar[i]:
            renk_sinifi = "status-on" if veri["durum"] == "Açık" else "status-off"
            deger_metni = f" ({veri['deger']})" if veri["deger"] is not None else ""
            
            st.markdown(f"""
            <div class="device-card">
                <h1 style="margin:0;">{veri['ikon']}</h1>
                <h4 style="margin:5px 0;">{cihaz_adi.capitalize()}</h4>
                <p class="{renk_sinifi}" style="margin:0;">{veri['durum']}{deger_metni}</p>
            </div>
            <br>
            """, unsafe_allow_html=True)