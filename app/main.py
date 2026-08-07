"""
app/main.py — Asistan Giris Noktasi
"""

import sys
import os

# Her durumda Smarthome/ dizinini path'e ekle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Artik app paketi bulunabilir
from app.core.asistan_dongu import baslat

baslat()
