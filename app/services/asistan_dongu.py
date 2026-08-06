"""
asistan_dongu.py — Akıllı Ev Ses Asistanı Döngüsü

Pipeline: Mikrofon → Wake Word → Dinamik VAD Kayıt → Whisper STT → asistan_beyin → MQTT → PiperTTS

Bileşenler:
  - openWakeWord: "Alexa" uyandırma kelimesi algılama
  - Adaptive VAD: Ortam gürültüsüne uyarlanabilir ses eşiği
  - Faster-Whisper: Türkçe STT (base model, int8, beam_size=1)
  - asistan_beyin: Hybrid Cache→Regex→LLM komut işleme
  - PiperTTS: Türkçe sesli geri bildirim (fahrettin-medium)

Kaynak: ai_test/asistan_dongu.py (izole test ortamından adapte edilmiştir)
"""

import time
import re
import sys
import asyncio
from collections import deque

import numpy as np
import sounddevice as sd
from openwakeword.utils import download_models
from openwakeword.model import Model
from faster_whisper import WhisperModel

# Asistan beyin modülü — komut metnini JSON'a çevirir
from app.services.asistan_beyin import komut_isle

# Sesli geri bildirim — PiperTTS ile Türkçe yanıt
from app.services.sesli_yanit import sesli_yanit

# MQTT servisi — sonucu IoT cihazlarına iletir
from app.core.mqtt import mqtt_client

# --- AYARLAR ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
BUFFER_SANIYE = 2
MAX_BUFFER_CHUNKS = int((SAMPLE_RATE / CHUNK_SIZE) * BUFFER_SANIYE)

print("Gerekli modeller kontrol ediliyor...")
download_models()

print("Faster-Whisper (Dinamik Uyarlanabilir VAD Modu) yükleniyor...")
whisper_model = WhisperModel(
    "base",  
    device="cpu",
    compute_type="int8",
    cpu_threads=6,
    num_workers=1
)

print("openWakeWord (Nöbetçi) yükleniyor...")
owwModel = Model(
    wakeword_models=["alexa"],
    inference_framework="onnx"
)

print("\n==================================================")
print("AKILLI EV ASİSTANI DİNLEMEDE (Gürültü Filtreli Dinamik Mod)...")
print("'Alexa' de, komutunu söyle ve normal şekilde dur. Kayıt anında bitecek!")
print("==================================================\n")


def _sonucu_mqtt_gonder(sonuc: dict):
    """
    AI komut sonucunu MQTT üzerinden IoT cihazlarına iletir.
    Eğer MQTT bağlantısı yoksa (standalone mod), sadece loglar.
    """
    if mqtt_client.client is None:
        print("  MQTT bağlantısı yok — standalone mod, sonuç sadece loglandı.")
        return

    # Async fonksiyonu sync context'ten çağır
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                mqtt_client.komut_gonder("ev/komutlar/ai", sonuc)
            )
        else:
            loop.run_until_complete(
                mqtt_client.komut_gonder("ev/komutlar/ai", sonuc)
            )
        print(f"  MQTT'ye iletildi: ev/komutlar/ai")
    except Exception as e:
        print(f"  MQTT gönderim hatası: {e}")


def basla():
    """
    Ana ses asistanı döngüsünü başlatır.
    Bu fonksiyon blocking'dir — ayrı bir process'te çalıştırılmalı.
    """
    try:
        audio_buffer = deque(maxlen=MAX_BUFFER_CHUNKS)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                audio_buffer.append(audio_data)
                owwModel.predict(audio_data)
                
                trigger_detected = False
                for model_name in owwModel.models.keys():
                    score = owwModel.prediction_buffer[model_name][-1]
                    if score > 0.3:
                        trigger_detected = True
                        break
                
                if trigger_detected:
                    print(f"\nUYANDIRMA YAKALANDI!")
                    print("Dinleniyor (Ortam gürültüsü analiz ediliyor ve komut bekleniyor)...")
                    
                    # Geçmişteki 2 saniyelik buffer'ı al
                    gecmis_ses = list(audio_buffer)
                    aktif_kayit_parcalari = []
                    
                    # UYARLANABİLİR EŞİK (Adaptive Threshold):
                    son_gurultu_ornekleri = np.concatenate(gecmis_ses).astype(np.float32)
                    ortam_gurultusu = np.sqrt(np.mean(son_gurultu_ornekleri**2))
                    SES_ESIGI = max(ortam_gurultusu * 1.5, 300) 
                    
                    MAKSIMUM_SESSIZLIK_CHUNK = 10 
                    
                    sessizlik_sayaci = 0
                    konusma_baslandi_mi = False
                    maksimum_guvenlik_suresi = 8 
                    max_dongu = int((SAMPLE_RATE / CHUNK_SIZE) * maksimum_guvenlik_suresi)
                    
                    for _ in range(max_dongu):
                        chunk, _ = stream.read(CHUNK_SIZE)
                        parca = np.frombuffer(chunk, dtype=np.int16)
                        aktif_kayit_parcalari.append(parca)
                        
                        rms = np.sqrt(np.mean(parca.astype(np.float32)**2))
                        
                        if rms > SES_ESIGI:
                            konusma_baslandi_mi = True
                            sessizlik_sayaci = 0 
                        elif konusma_baslandi_mi:
                            sessizlik_sayaci += 1
                            if sessizlik_sayaci >= MAKSIMUM_SESSIZLIK_CHUNK:
                                break
                    
                    gecmis_birlesik = np.concatenate(gecmis_ses)
                    ek_birlesik = np.concatenate(aktif_kayit_parcalari)
                    toplam_ses = np.concatenate((gecmis_birlesik, ek_birlesik))
                    
                    print("Ses tamamlandı, Whisper işliyor...")
                    
                    audio_fp32 = toplam_ses.astype(np.float32) / 32768.0
                    
                    t_start = time.time()
                    segments, info = whisper_model.transcribe(
                        audio_fp32, 
                        beam_size=1, 
                        language="tr",
                        initial_prompt="smart home assistant",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=400)
                    )
                    
                    komut_metni = ""
                    for segment in segments:
                        komut_metni += segment.text + " "
                    
                    komut_metni = komut_metni.strip()
                    t_end = time.time()
                    
                    # "Alexa", "Aleksa" ve "Aleks'a" varyasyonlarını noktalama işaretleriyle birlikte temizle
                    komut_metni = re.sub(r'(?i)\b(?:alexa|aleksa|aleks[\'']?a|alex[\'']?a)\b[,\.!?:;]*\s*', '', komut_metni).strip()

                    print(f"Çeviri Süresi: {t_end - t_start:.2f} saniye")
                    print(f"Algılanan Net Komut: \"{komut_metni}\"")
                    
                    # ============================================================
                    # KOMUT İŞLEME — asistan_beyin hybrid pipeline'ına gönder
                    # ============================================================
                    if komut_metni:
                        sonuc = komut_isle(komut_metni, etiket="SES_KOMUT")
                        
                        # Sonucu MQTT üzerinden IoT cihazlarına ilet
                        _sonucu_mqtt_gonder(sonuc)
                        
                        # Sesli geri bildirim — komut sonucuna göre
                        sesli_yanit.komut_sonucu(sonuc)
                    else:
                        print("  Boş komut algılandı, işleme atlandı.")
                        sesli_yanit.hata("bos_komut")
                    
                    print("\nTekrar nöbetçiye dönülüyor...\n")
                    owwModel.reset()
                    audio_buffer.clear()

    except KeyboardInterrupt:
        print("\nAsistan sistemi kapatıldı.")


# =============================================================================
# Standalone çalıştırma desteği:
#   python -m app.services.asistan_dongu
# =============================================================================
if __name__ == "__main__":
    basla()
