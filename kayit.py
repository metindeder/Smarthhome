"""
kayit.py — Biyometrik profil kayit / test script'i
Proje kokunden calistir:
    python kayit.py --kayit
    python kayit.py --test
"""
import sys
import os

# Smarthome/ dizinini path'e ekle
_KOK = os.path.dirname(os.path.abspath(__file__))
if _KOK not in sys.path:
    sys.path.insert(0, _KOK)

from app.core.biyometrik import kayit_modu, test_modu
import argparse

parser = argparse.ArgumentParser(description="Biyometrik Ses Profil Yoneticisi")
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("--kayit", action="store_true", help="Yeni profil kaydi (5 ornek)")
grp.add_argument("--test",  action="store_true", help="Mevcut profili test et")
args = parser.parse_args()

if args.kayit:
    kayit_modu()
elif args.test:
    test_modu()
