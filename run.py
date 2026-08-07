"""
run.py — Asistani baslatir.
Kullanim: python run.py
"""

import sys
import os

# Smarthome/ her zaman path'te olmali
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.asistan_dongu import baslat

baslat()
