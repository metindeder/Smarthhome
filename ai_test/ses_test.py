import os

# --- HUGGING FACE HIZLI İNDİRME AYARLARI ---
# Kendi HF token'ını buraya yapıştır (örnek: hf_AbCdEfG...)
os.environ["HF_TOKEN"] = "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" 

# Rust tabanlı ultra hızlı indirme motorunu aktif et 
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
# ------------------------------------------

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# --- AYARLAR ---
KAYIT_SURESI = 5  # Kaç saniye dinlenecek?
SAMPLE_RATE = 16000 # Whisper'ın varsayılan ses frekansı
MODEL_BOYUTU = "small" # CPU için en iyi hız/performans dengesi

print(f"🧠 Yapay Zeka Modeli ({MODEL_BOYUTU}) VIP hızında yükleniyor...")

# İşlemci dostu int8 optimizasyonu ile modeli CPU'da yüklüyoruz
model = WhisperModel(MODEL_BOYUTU, device="cpu", compute_type="int8")

print(f"\n🎙️ Lütfen konuşun (Mikrofon {KAYIT_SURESI} saniye boyunca dinleniyor)...")

# Ses kaydını başlat
kayit = sd.rec(int(KAYIT_SURESI * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait() # Kaydın bitmesini bekle
print("✅ Kayıt bitti, yapay zeka sesi çözümlüyor...")

# Sesi geçici bir WAV dosyasına kaydet
gecici_dosya = "test_sesi.wav"
wav.write(gecici_dosya, SAMPLE_RATE, kayit)

# Sesi metne dök 
segments, info = model.transcribe(gecici_dosya, beam_size=5, language="tr")

print(f"\n--- SONUÇ (Algılanan Dil: {info.language}) ---")
for segment in segments:
    print(f"🗣️ Sen: {segment.text.strip()}")
print("---------------------------------------------")

# Geçici dosyayı sil
if os.path.exists(gecici_dosya):
    os.remove(gecici_dosya)