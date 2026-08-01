import numpy as np
import sounddevice as sd
from openwakeword.utils import download_models
from openwakeword.model import Model

print("📥 Eksik openWakeWord modelleri kontrol ediliyor ve indiriliyor...")
# Eksik model dosyalarını resmi GitHub reposundan indirip doğru klasöre yerleştirir
download_models()

print("🦉 openWakeWord ONNX nöbetçisi yükleniyor...")
owwModel = Model(
    wakeword_models=["alexa"],
    inference_framework="onnx"
)

CHUNK_SIZE = 1280  
SAMPLE_RATE = 16000

print("\n👂 Nöbetçi kulak devrede! Lütfen net bir şekilde 'Alexa' de...")
print("(Çıkış için CTRL+C yapabilirsin)")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
        while True:
            audio_chunk, overflowed = stream.read(CHUNK_SIZE)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            prediction = owwModel.predict(audio_data)
            
            for model_name in owwModel.models.keys():
                score = owwModel.prediction_buffer[model_name][-1]
                if score > 0.5:
                    print(f"\n🚨 UYANDIRMA KELİMESİ YAKALANDI! (Skor: {score:.2f})")
                    print("🎙️ Şimdi Faster-Whisper'ı tetikleyebiliriz...")

except KeyboardInterrupt:
    print("\n🛑 Nöbetçi durduruldu.")