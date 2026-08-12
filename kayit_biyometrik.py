"""
kayit_biyometrik.py — Tamamen bagimsiz biyometrik profil kayit araci
Hicbir proje paketi import etmez. Direk calistir:

    python kayit_biyometrik.py
    python kayit_biyometrik.py --test
"""
import os
import sys

# Yerel model dosyalari varsa tam offline modda calis (ag baglanma)
_MODEL_DIZIN_KONTROL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "spkrec-ecapa-voxceleb", "hyperparams.yaml"
)
if os.path.exists(_MODEL_DIZIN_KONTROL):
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    # Yerel dosya yok, mirror'dan indir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
import argparse
import numpy as np

# ---------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------
# Profil bu dosyanin yanindaki 'data/' klasorune kaydedilir
PROJE_KOK      = os.path.dirname(os.path.abspath(__file__))
PROFIL_DOSYA   = os.path.join(PROJE_KOK, "data", "sahip_profili.pt")
MODEL_DIZIN    = os.path.join(PROJE_KOK, "models", "spkrec-ecapa-voxceleb")

SAMPLE_RATE        = 16000
KAYIT_SURE_SN      = 3
KAYIT_ORNEK_SAYISI = 5
ESIK               = 0.55   # Dogrulama esigi

ORNEK_CUMLELER = [
    "Alexa, lambayı aç",
    "Alexa, klimayı kapat",
    "Alexa, müziği durdur",
    "Alexa, perdeyi indir",
    "Alexa, salon ışıklarını aç",
]

# ---------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# ---------------------------------------------------------------

def _cuda_var_mi() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _model_yukle():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("\n[HATA] speechbrain kurulu degil!")
        print("Kurmak icin:  pip install speechbrain")
        sys.exit(1)

    cihaz = "cuda" if _cuda_var_mi() else "cpu"
    print(f"  ECAPA-TDNN model yukleniyor (device={cihaz})...")
    t0 = time.perf_counter()

    # Yerel dosya var mi? (daha once indirildi mi)
    _yerel_var = os.path.exists(os.path.join(MODEL_DIZIN, "hyperparams.yaml"))
    enc = None

    if _yerel_var:
        try:
            print("  Yerel onbellekten yukleniyor...")
            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=MODEL_DIZIN,
                run_opts={"device": cihaz},
                local_files_only=True,
            )
        except Exception:
            pass

    if enc is None:
        print(f"  HF mirror'dan indiriliyor ({os.environ.get('HF_ENDPOINT', 'hf-mirror.com')})...")
        try:
            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=MODEL_DIZIN,
                run_opts={"device": cihaz},
            )
        except Exception as e:
            print(f"\n[HATA] Model indirilemedi: {e}")
            print("\nManuel indirme adımlari:")
            print("  1. Tarayicinizdan su adrese gidin:")
            print("     https://hf-mirror.com/speechbrain/spkrec-ecapa-voxceleb/tree/main")
            print("  2. Asagidaki 3 dosyayi indirin:")
            print("     - hyperparams.yaml")
            print("     - embedding_model.ckpt")
            print("     - mean_var_norm_emb.ckpt")
            print(f"  3. Indirilen dosyalari su klasore koyun:")
            print(f"     {MODEL_DIZIN}")
            print("  4. Tekrar calistirin.")
            sys.exit(1)

    ms = (time.perf_counter() - t0) * 1000
    print(f"  Model hazir! ({ms:.0f} ms)\n")
    return enc


def _embedding_cikart(enc, audio_fp32: np.ndarray):
    import torch
    import torch.nn.functional as F

    tensor   = torch.from_numpy(audio_fp32).float().unsqueeze(0)   # (1, N)
    wav_lens = torch.ones(1)

    with torch.no_grad():
        emb = enc.encode_batch(tensor, wav_lens)

    emb = emb.squeeze()
    emb = F.normalize(emb, dim=0)
    return emb


def _profil_kaydet(emb):
    import torch
    os.makedirs(os.path.dirname(PROFIL_DOSYA), exist_ok=True)
    torch.save(emb, PROFIL_DOSYA)


def _profil_yukle():
    if not os.path.exists(PROFIL_DOSYA):
        return None
    import torch
    return torch.load(PROFIL_DOSYA, weights_only=True)


def _cosine(a, b) -> float:
    import torch
    return torch.dot(a.cpu(), b.cpu()).item()


def _mikrofon_kaydet(sure_sn: int) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("[HATA] sounddevice kurulu degil:  pip install sounddevice")
        sys.exit(1)

    ses = sd.rec(
        int(sure_sn * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return ses.squeeze()


# DeepFilterNet — runtime ile ayni pipeline (lazy-load)
_df_model = None
_df_state = None
_df_sr    = None


def _gurultuden_arindir(audio_f32: np.ndarray) -> np.ndarray:
    """Runtime'daki gibi DeepFilterNet uygula. Yuklenemezse ham sesi dondur."""
    global _df_model, _df_state, _df_sr
    if _df_model is None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from df.enhance import init_df
            _df_model, _df_state, _ = init_df()
            _df_sr = _df_state.sr()
            print(f"  DeepFilterNet yuklendi (SR={_df_sr} Hz), ses temizleniyor...")
        except Exception:
            print("  [UYARI] DeepFilterNet yuklenemedi, ham ses kullanilacak.")
            _df_model = False   # tekrar deneme
    if not _df_model:
        return audio_f32
    try:
        import torch, torchaudio, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from df.enhance import enhance
        tensor = torch.from_numpy(audio_f32).float().unsqueeze(0)
        if SAMPLE_RATE != _df_sr:
            tensor = torchaudio.functional.resample(tensor, SAMPLE_RATE, _df_sr)
        enhanced = enhance(_df_model, _df_state, tensor)
        if SAMPLE_RATE != _df_sr:
            enhanced = torchaudio.functional.resample(enhanced, _df_sr, SAMPLE_RATE)
        return enhanced.squeeze(0).numpy()
    except Exception:
        return audio_f32



# ---------------------------------------------------------------
# KAYIT MODU
# ---------------------------------------------------------------

def kayit_modu():
    print("=" * 55)
    print("  BIYOMETRIK KAYIT MODU")
    print(f"  {KAYIT_ORNEK_SAYISI} farkli cumle kayit edilecek.")
    print(f"  Her kayit {KAYIT_SURE_SN} saniye surer.")
    print("=" * 55)

    enc = _model_yukle()
    embeddingler = []

    for i, cumle in enumerate(ORNEK_CUMLELER[:KAYIT_ORNEK_SAYISI]):
        print(f"  [{i+1}/{KAYIT_ORNEK_SAYISI}]  Soyle: \"{cumle}\"")
        for geri in range(3, 0, -1):
            print(f"  {geri}...", end="\r", flush=True)
            time.sleep(1.0)
        print(f"  [KAYIT!]          ")

        audio = _mikrofon_kaydet(KAYIT_SURE_SN)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            print(f"  Ses cok sessiz (RMS={rms:.4f}), atlandi.\n")
            continue

        # Runtime ile ayni pipeline: DeepFilterNet uygula
        audio = _gurultuden_arindir(audio)

        emb = _embedding_cikart(enc, audio)
        embeddingler.append(emb)
        print(f"  Kaydedildi. (RMS={rms:.4f})\n")


    if len(embeddingler) < 2:
        print("HATA: Yeterli ornek alinamadi, kayit iptal.")
        return

    import torch, torch.nn.functional as F
    ortalama = torch.stack(embeddingler).mean(dim=0)
    ortalama = F.normalize(ortalama, dim=0)

    _profil_kaydet(ortalama)
    print("=" * 55)
    print(f"  Profil olusturuldu! ({len(embeddingler)} ornek)")
    print(f"  Dosya: {PROFIL_DOSYA}")
    print("=" * 55)


# ---------------------------------------------------------------
# TEST MODU
# ---------------------------------------------------------------

def test_modu():
    print("=" * 55)
    print("  BIYOMETRIK TEST MODU")
    print("=" * 55)

    profil = _profil_yukle()
    if profil is None:
        print(f"Profil bulunamadi: {PROFIL_DOSYA}")
        print("Once '--kayit' ile profil olusturun.")
        return

    enc = _model_yukle()
    print("  3 saniye konusin...")
    time.sleep(1)
    print("  [KAYIT!]")

    audio = _mikrofon_kaydet(3)
    emb   = _embedding_cikart(enc, audio)
    skor  = _cosine(emb, profil)

    kabul = skor >= ESIK
    print(f"\n  Skor  : {skor:.4f}")
    print(f"  Esik  : {ESIK}")
    print(f"  Sonuc : {'YETKILI  (gecti)' if kabul else 'YETKISIZ (reddedildi)'}")
    print("=" * 55)


# ---------------------------------------------------------------
# GIRIS NOKTASI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Biyometrik Ses Profil Araci",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ornek kullanim:\n"
            "  python kayit_biyometrik.py          # kayit\n"
            "  python kayit_biyometrik.py --test   # test\n"
        ),
    )
    parser.add_argument("--test", action="store_true", help="Mevcut profili test et")
    args = parser.parse_args()

    if args.test:
        test_modu()
    else:
        kayit_modu()
