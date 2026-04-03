# src/__init__.py

# Expose kelas agar bisa diimport langsung dari package src
from .encoder import SelfEncoder
from .predictor import SelfPredictor
from .decoder import ActionDecoder
from .memory import RecursiveCore # (Asumsi memory.py juga ada di folder src)

# List modul apa saja yang tersedia jika seseorang melakukan 'from src import *'
__all__ = [
    "SelfEncoder",
    "SelfPredictor",
    "ActionDecoder",
    "RecursiveCore"
]