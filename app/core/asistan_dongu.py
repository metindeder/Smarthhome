"""
app/core/asistan_dongu.py

ai_test/asistan_dongu.py (ses dinleme + wake-word + Whisper)
ile ai_test/asistan_beyin.py (komut isleme) arasindaki kopru.

Akis:
    [Wake Word] -> [DeepFilterNet] -> [Whisper] -> komut_metni -> komut_isle()
"""

import time
import re
from collections import deque

import numpy as np
import sounddevice as sd
from openwakeword.utils import download_models
from openwakeword.model import Model
from faster_whisper import WhisperModel

from .asistan_beyin import komut_isle
from .biyometrik import dogrula as biyometrik_dogrula

# --- DeepFilterNet (lazy-load: ilk kullanımda yüklenir) ---
_df_model = None
_df_state = None
_df_sr = None


def _get_df():
    """DeepFilterNet modelini bir kez yukler ve onbellekte tutar."""
    global _df_model, _df_state, _df_sr
    if _df_model is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from df.enhance import init_df  # noqa: F401
        # init_df() -> (model, df_state, model_name_str)
        # Gercek sample rate: df_state.sr()
        _df_model, _df_state, _ = init_df()
        _df_sr = _df_state.sr()
        print(f"DeepFilterNet yuklendi (model SR={_df_sr} Hz)")
    return _df_model, _df_state, _df_sr


def gurultuden_arindir(audio_int16: np.ndarray, giris_sr: int = 16000) -> np.ndarray:
    """
    int16 ses dizisini DeepFilterNet ile temizler.
    Donus: giris_sr oraninda float32 [-1, 1] dizi.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from df.enhance import enhance
        import torch
        import torchaudio

    df_model, df_state, df_sr = _get_df()

    # int16 -> float32 [-1, 1]
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # DF modeli stereo/mono tensor bekler: (kanal, ornek)
    tensor = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, N)

    # Giris SR != DF SR ise resample et
    if giris_sr != df_sr:
        tensor = torchaudio.functional.resample(tensor, giris_sr, df_sr)

    # Gurultu arindirma
    enhanced = enhance(df_model, df_state, tensor)  # (1, N) veya (N,)

    # Cikis SR -> giris SR'e geri resample
    if giris_sr != df_sr:
        enhanced = torchaudio.functional.resample(enhanced, df_sr, giris_sr)

    return enhanced.squeeze(0).numpy()

# --- AYARLAR (ai_test/asistan_dongu.py ile ayni) ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
BUFFER_SANIYE = 2
MAX_BUFFER_CHUNKS = int((SAMPLE_RATE / CHUNK_SIZE) * BUFFER_SANIYE)

# Alexa ve varyasyonlarini temizleyen regex (onceden derlendi)
_ALEXA_RE = re.compile(
    r"(?i)\b(?:alexa|aleksa|aleks['\u2019]?a|alex['\u2019]?a)\b[,\.!?:;]*\s*"
)


def baslat():
    print("Gerekli modeller kontrol ediliyor...")
    download_models()

    print("Faster-Whisper yukleniyor...")
    whisper_model = WhisperModel(
        "base", device="cpu", compute_type="int8", cpu_threads=6, num_workers=1
    )

    print("openWakeWord yukleniyor...")
    owwModel = Model(wakeword_models=["alexa"], inference_framework="onnx")

    print("\n==================================================")
    print("AKILLI EV ASISTANI DINLEMEDE...")
    print("'Alexa' de, komutunu soyle ve dur.")
    print("==================================================\n")

    try:
        audio_buffer = deque(maxlen=MAX_BUFFER_CHUNKS)

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK_SIZE
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_buffer.append(audio_data)
                owwModel.predict(audio_data)

                tetik = any(
                    owwModel.prediction_buffer[name][-1] > 0.3
                    for name in owwModel.models.keys()
                )

                if tetik:
                    print("\nUYANDIRMA YAKALANDI! Dinleniyor...")

                    gecmis_ses = list(audio_buffer)
                    aktif_kayit_parcalari = []

                    son_gurultu = np.concatenate(gecmis_ses).astype(np.float32)
                    SES_ESIGI = max(np.sqrt(np.mean(son_gurultu ** 2)) * 1.5, 300)

                    sessizlik_sayaci = 0
                    konusma_basladi = False
                    max_dongu = int((SAMPLE_RATE / CHUNK_SIZE) * 8)

                    for _ in range(max_dongu):
                        chunk, _ = stream.read(CHUNK_SIZE)
                        parca = np.frombuffer(chunk, dtype=np.int16)
                        aktif_kayit_parcalari.append(parca)
                        rms = np.sqrt(np.mean(parca.astype(np.float32) ** 2))
                        if rms > SES_ESIGI:
                            konusma_basladi = True
                            sessizlik_sayaci = 0
                        elif konusma_basladi:
                            sessizlik_sayaci += 1
                            if sessizlik_sayaci >= 10:
                                break

                    toplam_ses = np.concatenate([
                        np.concatenate(gecmis_ses),
                        np.concatenate(aktif_kayit_parcalari),
                    ])

                    print("Ses tamamlandi, DeepFilterNet ile gurultu arindiriliyor...")
                    t_df_start = time.perf_counter()
                    audio_fp32 = gurultuden_arindir(toplam_ses, giris_sr=SAMPLE_RATE)
                    df_ms = (time.perf_counter() - t_df_start) * 1000
                    print(f"Gurultu arindirma tamamlandi ({df_ms:.0f}ms), Whisper isliyor...")

                    # --- BİYOMETRİK DOĞRULAMA ---
                    if not biyometrik_dogrula(audio_fp32, sr=SAMPLE_RATE):
                        print("  [BİYOMETRİK] Yetkisiz ses! Komut reddedildi.")
                        print("\nTekrar nobetciye donuluyor...\n")
                        owwModel.reset()
                        audio_buffer.clear()
                        continue

                    t_start = time.perf_counter()
                    segments, _ = whisper_model.transcribe(
                        audio_fp32,
                        beam_size=1,
                        language="tr",
                        initial_prompt="smart home assistant",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=400),
                    )
                    komut_metni = "".join(seg.text + " " for seg in segments).strip()
                    komut_metni = _ALEXA_RE.sub("", komut_metni).strip()
                    
                    stt_ms = (time.perf_counter() - t_start) * 1000
                    stt_sure_sn = stt_ms / 1000.0
                    print(f"Ceviri ({stt_sure_sn:.2f}s) -> '{komut_metni}'")

                    if komut_metni:
                        komut_isle(komut_metni, etiket="CANLI", stt_ms=stt_ms)

                    print("\nTekrar nobetciye donuluyor...\n")
                    owwModel.reset()
                    audio_buffer.clear()

    except KeyboardInterrupt:
        print("\nAsistan durduruldu.")


if __name__ == "__main__":
    baslat()
