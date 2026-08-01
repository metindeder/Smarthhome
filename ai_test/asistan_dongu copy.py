import os

import time

import re

from collections import deque

import numpy as np

import sounddevice as sd

import scipy.io.wavfile as wav

from openwakeword.utils import download_models

from openwakeword.model import Model

from faster_whisper import WhisperModel



# --- AYARLAR ---

SAMPLE_RATE = 16000

CHUNK_SIZE = 1280

# 2 saniyelik geçmiş sesi hafızada tutmak için buffer boyutu:

BUFFER_SANIYE = 2

MAX_BUFFER_CHUNKS = int((SAMPLE_RATE / CHUNK_SIZE) * BUFFER_SANIYE)



print("📥 Gerekli modeller kontrol ediliyor...")

download_models()



print("🧠 Faster-Whisper (STT) yükleniyor...")

# OPTİMİZASYON 1: Thread sayısını sınırlayarak bağlam geçişini önlüyoruz

whisper_model = WhisperModel(

    "small",  # Çok daha hızlı olmasını istersen burayı "base" yapabilirsin

    device="cpu",

    compute_type="int8",

    cpu_threads=6,

    num_workers=1

)



print("🦉 openWakeWord (Nöbetçi) yükleniyor...")

owwModel = Model(

    wakeword_models=["alexa"],

    inference_framework="onnx"

)



print("\n==================================================")

print("🎙️ AKILLI EV ASİSTANI DİNLEMEDE (Buffer Aktif)...")

print("👉 Tek nefeste söyleyebilirsin: 'Alexa, ışıkları aç'")

print("==================================================\n")



try:

    audio_buffer = deque(maxlen=MAX_BUFFER_CHUNKS)



    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:

        while True:

            audio_chunk, overflowed = stream.read(CHUNK_SIZE)

            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

           

            # Sesi sürekli hafızada tut

            audio_buffer.append(audio_data)

           

            # Nöbetçi kontrolü

            owwModel.predict(audio_data)

           

            trigger_detected = False

            for model_name in owwModel.models.keys():

                score = owwModel.prediction_buffer[model_name][-1]

                if score > 0.4:

                    trigger_detected = True

                    break

           

            if trigger_detected:

                print(f"\n🚨 UYANDIRMA YAKALANDI!")

                print("🎙️ Kayıt alınıyor...")

               

                # Hafızadaki son 2 saniyelik sesi ve üstüne gelecek 3 saniyeyi birleştir

                ek_kayit_suresi = 3

                ek_kayit = sd.rec(int(ek_kayit_suresi * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

                sd.wait()

               

                gecmis_ses = np.concatenate(list(audio_buffer))

                toplam_ses = np.concatenate((gecmis_ses, ek_kayit.flatten()))

               

                print("✅ Ses alındı, yapay zeka çeviriyor...")

               

                gecici_dosya = "komut_gecici.wav"

                wav.write(gecici_dosya, SAMPLE_RATE, toplam_ses)

               

                # OPTİMİZASYON 2 & 3: Beam_size=1 ile Greedy Search ve VAD Filtresi

                t_start = time.time()

                segments, info = whisper_model.transcribe(

                    gecici_dosya,

                    beam_size=1,

                    language="tr",

                    vad_filter=True,

                    vad_parameters=dict(min_silence_duration_ms=500)

                )

               

                komut_metni = ""

                for segment in segments:

                    komut_metni += segment.text + " "

               

                komut_metni = komut_metni.strip()

                t_end = time.time()

               

               # "Alexa", "Aleksa" (büyük/küçük harf fark etmeksizin) ve noktalama işaretlerini temizle

                komut_metni = re.sub(r'(?i)\b(?:alexa|aleksa)\b[,\.!?:;]*\s*', '', komut_metni).strip()



                print(f"⏱️ Çeviri Süresi: {t_end - t_start:.2f} saniye")

                print(f"🗣️ Algılanan Net Komut: \"{komut_metni}\"")

               

                if os.path.exists(gecici_dosya):

                    os.remove(gecici_dosya)

                   

                print("\n🔄 Tekrar nöbetçiye dönülüyor...\n")

                owwModel.reset()

                audio_buffer.clear()



except KeyboardInterrupt:

    print("\n🛑 Asistan sistemi kapatıldı.")