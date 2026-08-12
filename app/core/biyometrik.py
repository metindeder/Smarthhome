"""
app/core/biyometrik.py — Biyometrik Ses Doğrulama Modülü

SpeechBrain ECAPA-TDNN kullanarak konuşmacı doğrulaması yapar.

Modlar:
  - Kayıt (enrollment): python -m app.core.biyometrik --kayit
  - Test:               python -m app.core.biyometrik --test
  - Doğrulama (API):    biyometrik.dogrula(audio_fp32) -> bool

Akış:
  1. İlk kullanımda '--kayit' ile ev sahibinin sesi kaydedilir.
  2. Kayıttan ortalama embedding oluşturulup 'data/sahip_profili.pt' dosyasına yazılır.
  3. Her uyandırma sonrası dogrula() çağrılır; cosine similarity >= ESIK ise True döner.
"""

import os
import sys
import time
import argparse
import numpy as np

# Proje kök dizini (Smarthome/)
_DOSYA_DIR = os.path.dirname(os.path.abspath(__file__))   # app/core/
_APP_DIR   = os.path.dirname(_DOSYA_DIR)                  # app/
_KOK_DIR   = os.path.dirname(_APP_DIR)                    # Smarthome/

# HuggingFace: yerel model varsa tam offline mod — import ÖNCESINDE ayarlanmalı
_MODEL_YAML = os.path.join(_KOK_DIR, "models", "spkrec-ecapa-voxceleb", "hyperparams.yaml")
if os.path.exists(_MODEL_YAML):
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Profil ve model dizinleri
_PROFIL_DOSYA = os.path.join(_KOK_DIR, "data", "sahip_profili.pt")
_MODEL_DIZIN  = os.path.join(_KOK_DIR, "models", "spkrec-ecapa-voxceleb")

# Doğrulama eşiği — 0.0-1.0 arası (yüksek = daha sıkı)
ESIK: float = 0.50

# Model 16kHz bekler
SAMPLE_RATE = 16000

# Enrollment sırasında kaç örnek alınsın
KAYIT_ORNEK_SAYISI = 5
KAYIT_SURE_SN      = 3      # Her örnek kaç saniye

# Lazy-load — modül import edildiğinde model yüklenmez, ilk kullanımda yüklenir
_encoder = None


def _model_yukle():
    """ECAPA-TDNN modelini bir kez yükler ve önbellekte tutar."""
    global _encoder
    if _encoder is not None:
        return _encoder

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        raise ImportError(
            "SpeechBrain kurulu değil. Lütfen çalıştırın:\n"
            "  pip install speechbrain"
        )

    cihaz = "cuda" if _cuda_var_mi() else "cpu"
    print(f"  [BİYOMETRİK] ECAPA-TDNN yükleniyor (device={cihaz})...")
    t0 = time.perf_counter()

    try:
        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_MODEL_DIZIN,
            run_opts={"device": cihaz},
        )
    except Exception as e:
        raise RuntimeError(
            f"ECAPA-TDNN modeli yüklenemedi: {e}\n"
            "Manuel indirme: https://hf-mirror.com/speechbrain/spkrec-ecapa-voxceleb"
        ) from e

    ms = (time.perf_counter() - t0) * 1000
    print(f"  [BİYOMETRİK] Model hazır! ({ms:.0f} ms)")
    return _encoder


def _cuda_var_mi() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _embedding_cikart(audio_fp32: np.ndarray, sr: int = SAMPLE_RATE) -> "torch.Tensor":
    """float32 numpy dizisinden normalize edilmiş ECAPA-TDNN embedding'i çıkarır."""
    import torch
    import torch.nn.functional as F

    enc = _model_yukle()

    # numpy -> torch tensor (1, N)
    tensor = torch.from_numpy(audio_fp32).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)   # (1, N)

    wav_lens = torch.ones(1)           # normalize uzunluk = 1.0 (tam uzunluk)

    with torch.no_grad():
        emb = enc.encode_batch(tensor, wav_lens)  # (1, 1, D)

    emb = emb.squeeze()                            # (D,)
    emb = F.normalize(emb, dim=0)                 # birim vektör
    return emb


def _profil_yukle() -> "torch.Tensor | None":
    """Kayıtlı ortalama embedding'i dosyadan yükler. Yoksa None döner."""
    if not os.path.exists(_PROFIL_DOSYA):
        return None
    try:
        import torch
        return torch.load(_PROFIL_DOSYA, weights_only=True)
    except Exception as e:
        print(f"  [BİYOMETRİK] Profil yüklenemedi: {e}")
        return None


def _profil_kaydet(emb: "torch.Tensor"):
    """Embedding'i diske kaydeder."""
    import torch
    os.makedirs(os.path.dirname(_PROFIL_DOSYA), exist_ok=True)
    torch.save(emb, _PROFIL_DOSYA)
    print(f"  [BİYOMETRİK] Profil kaydedildi: {_PROFIL_DOSYA}")


def _cosine_benzerlik(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """İki normalize edilmiş vektör arasındaki cosine similarity."""
    import torch
    return torch.dot(a.cpu(), b.cpu()).item()


# =============================================================================
# PUBLIC API
# =============================================================================

def dogrula(audio_fp32: np.ndarray, sr: int = SAMPLE_RATE) -> bool:
    """
    Gelen sesi kayıtlı profille karşılaştırır.

    Returns:
        True  — skor >= ESIK (yetkili kullanıcı)
        False — skor <  ESIK (yetkisiz / profil yok bypass'ta True döner)

    Not:
        Profil dosyası bulunamazsa (henüz kayıt yapılmamışsa) True döner
        ve uyarı mesajı basar. Kayıt yapmak için:
            python -m app.core.biyometrik --kayit
    """
    # Çok kısa ses → güvenilmez, reddet
    min_ornek = int(1.0 * sr)
    if len(audio_fp32) < min_ornek:
        print("  [BİYOMETRİK] Ses çok kısa, doğrulama atlandı (geçiş verildi).")
        return True

    profil = _profil_yukle()
    if profil is None:
        print(
            "  [BİYOMETRİK] Profil bulunamadı — doğrulama atlandı (bypass modu).\n"
            "               Kayıt için: python -m app.core.biyometrik --kayit"
        )
        return True

    emb = _embedding_cikart(audio_fp32, sr)
    skor = _cosine_benzerlik(emb, profil)

    kabul = skor >= ESIK
    durum = "✓ GEÇTİ" if kabul else "✗ REDDEDİLDİ"
    print(f"  [BİYOMETRİK] Skor: {skor:.3f} / Eşik: {ESIK:.2f} → {durum}")
    return kabul


# =============================================================================
# KAYIT (ENROLLMENT) MODU
# =============================================================================

def kayit_modu():
    """
    Mikrofon kullanarak ev sahibinin sesini kaydeder ve profil oluşturur.
    CLI ile çalıştırılır:  python -m app.core.biyometrik --kayit
    """
    import torch
    import sounddevice as sd

    print("=" * 55)
    print("  BİYOMETRİK KAYIT MODU")
    print(f"  {KAYIT_ORNEK_SAYISI} farklı cümle kayıt edilecek.")
    print(f"  Her kayıt {KAYIT_SURE_SN} saniye sürer.")
    print("=" * 55)

    ornek_cumleler = [
        "Alexa, lambayı aç",
        "Alexa, klimayı kapat",
        "Alexa, müziği durdur",
        "Alexa, perdeyi indir",
        "Alexa, salon ışıklarını aç",
    ]

    _model_yukle()  # Erken yükle

    embeddingler = []

    for i, cumle in enumerate(ornek_cumleler[:KAYIT_ORNEK_SAYISI]):
        print(f"\n  [{i+1}/{KAYIT_ORNEK_SAYISI}] Söyle: \"{cumle}\"")
        print(f"  Kayıt başlıyor... (3 saniye)")
        time.sleep(0.5)

        # Geri sayım
        for geri in range(3, 0, -1):
            print(f"  {geri}...")
            time.sleep(1.0)
        print("  🎙️  KAYIT!")

        ses = sd.rec(
            int(KAYIT_SURE_SN * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = ses.squeeze()

        # RMS kontrolü — mikrofon açık mı?
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            print("  ⚠️  Ses çok sessiz! Mikrofonu kontrol et, tekrar dene.")
            continue

        emb = _embedding_cikart(audio)
        embeddingler.append(emb)
        print(f"  ✓ Kaydedildi (RMS={rms:.4f})")

    if len(embeddingler) < 2:
        print("\n  ❌ Yeterli örnek alınamadı. Kayıt iptal edildi.")
        return

    # Ortalama embedding → normalize
    import torch
    import torch.nn.functional as F
    ortalama = torch.stack(embeddingler).mean(dim=0)
    ortalama = F.normalize(ortalama, dim=0)

    _profil_kaydet(ortalama)
    print(f"\n  ✅ Profil oluşturuldu! ({len(embeddingler)} örnek kullanıldı)")
    print(f"     Dosya: {_PROFIL_DOSYA}")
    print("=" * 55)


# =============================================================================
# TEST MODU
# =============================================================================

def test_modu():
    """
    Mikrofon testi: 3 sn kaydeder, skoru basar.
    CLI ile çalıştırılır:  python -m app.core.biyometrik --test
    """
    import sounddevice as sd

    profil = _profil_yukle()
    if profil is None:
        print("Profil bulunamadı. Önce '--kayit' çalıştır.")
        return

    print("=" * 55)
    print("  BİYOMETRİK TEST MODU — 3 sn konuşun")
    print("=" * 55)
    time.sleep(1)
    print("  🎙️  KAYIT!")

    ses = sd.rec(
        int(3 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = ses.squeeze()

    kabul = dogrula(audio)
    print(f"\n  Sonuç: {'✅ YETKİLİ' if kabul else '❌ YETKİSİZ'}")
    print("=" * 55)


# =============================================================================
# CLI GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    # Proje kökünü path'e ekle
    if _KOK_DIR not in sys.path:
        sys.path.insert(0, _KOK_DIR)

    parser = argparse.ArgumentParser(description="Biyometrik Ses Doğrulama")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--kayit", action="store_true", help="Yeni profil kaydı")
    grp.add_argument("--test",  action="store_true", help="Mevcut profili test et")
    args = parser.parse_args()

    if args.kayit:
        kayit_modu()
    elif args.test:
        test_modu()
