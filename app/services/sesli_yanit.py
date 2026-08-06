"""
sesli_yanit.py — PiperTTS ile Sesli Geri Bildirim Servisi

Komut işlendikten sonra kullanıcıya sesli yanıt verir.
Önceden tanımlı mesajlar ile hızlı TTS — WAV buffer'dan doğrudan çalınır.

Kullanım:
    from app.services.sesli_yanit import sesli_yanit
    sesli_yanit.basarili("TV")
    sesli_yanit.hata("baglanti")
"""

import io
import wave
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

# =============================================================================
# 1. MODEL YÜKLEME
# =============================================================================
_PROJE_KOK = Path(__file__).resolve().parent.parent.parent
_MODEL_YOLU = str(_PROJE_KOK / "models" / "tts" / "tr_TR-fahrettin-medium.onnx")

print("PiperTTS yükleniyor...")
_t0 = time.perf_counter()
_voice = PiperVoice.load(_MODEL_YOLU)
_yuklenme_ms = (time.perf_counter() - _t0) * 1000
print(f"PiperTTS hazır! ({_yuklenme_ms:.0f} ms)")

# =============================================================================
# 2. SES ÇALMA — WAV buffer'dan sounddevice ile doğrudan çalma
# =============================================================================
def _sentezle_ve_cal(metin: str):
    """Metni sese çevirir ve hoparlörden çalar (blocking)."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        _voice.synthesize_wav(metin, wav_file)

    wav_buffer.seek(0)
    with wave.open(wav_buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()


def _cal_arka_planda(metin: str):
    """TTS'i ayrı thread'de çalıştırır — ana döngüyü bloke etmez."""
    t = threading.Thread(target=_sentezle_ve_cal, args=(metin,), daemon=True)
    t.start()
    return t

# =============================================================================
# 3. ÖN TANIMLI MESAJLAR
# =============================================================================
_MESAJLAR = {
    # Başarı mesajları
    "basarili":     "Komut başarıyla uygulandı.",
    "cihaz_acildi": "{cihaz} açıldı.",
    "cihaz_kapandi":"{cihaz} kapatıldı.",
    "ayar_yapildi": "{cihaz} ayarlandı.",

    # Hata mesajları
    "anlasilamadi":  "Komutu anlayamadım, lütfen tekrar söyleyin.",
    "baglanti":      "Bağlantı hatası oluştu.",
    "bos_komut":     "Herhangi bir komut algılanamadı.",
}


class SesliYanit:
    """Komut sonrası sesli geri bildirim servisi."""

    def soyle(self, metin: str, bekle: bool = False):
        """
        Verilen metni seslendirir.
        
        Args:
            metin: Söylenecek metin
            bekle: True ise ses bitene kadar bekler
        """
        print(f"  TTS: \"{metin}\"")
        if bekle:
            _sentezle_ve_cal(metin)
        else:
            _cal_arka_planda(metin)

    def basarili(self, cihaz: str = None, aksiyon: str = None):
        """Başarılı komut geri bildirimi."""
        if cihaz and aksiyon:
            if aksiyon in ("ac", "baslat"):
                metin = _MESAJLAR["cihaz_acildi"].format(cihaz=cihaz)
            elif aksiyon == "kapat":
                metin = _MESAJLAR["cihaz_kapandi"].format(cihaz=cihaz)
            elif aksiyon in ("ayarla", "artir", "azalt"):
                metin = _MESAJLAR["ayar_yapildi"].format(cihaz=cihaz)
            else:
                metin = _MESAJLAR["basarili"]
        else:
            metin = _MESAJLAR["basarili"]
        self.soyle(metin)

    def hata(self, tip: str = "anlasilamadi"):
        """Hata geri bildirimi."""
        metin = _MESAJLAR.get(tip, _MESAJLAR["anlasilamadi"])
        self.soyle(metin)

    def komut_sonucu(self, sonuc: dict):
        """
        AI pipeline sonucuna göre otomatik sesli yanıt.
        komut_isle() çıktısını direkt verebilirsiniz.
        """
        cihaz = sonuc.get("cihaz", "")
        aksiyon = sonuc.get("aksiyon", "")

        if cihaz == "bilinmiyor" or aksiyon == "bilinmiyor":
            self.hata("anlasilamadi")
        else:
            self.basarili(cihaz=cihaz, aksiyon=aksiyon)


# Singleton instance — dışarıdan import edilir
sesli_yanit = SesliYanit()
