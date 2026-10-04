"""Singleton predictor loaded once at API startup."""
from __future__ import annotations

import os
from functools import lru_cache

from src.predict import DefectPredictor


@lru_cache(maxsize=1)
def get_predictor() -> DefectPredictor:
    ckpt = os.environ.get("MODEL_PATH", "models/best_model.pt")
    return DefectPredictor(ckpt)