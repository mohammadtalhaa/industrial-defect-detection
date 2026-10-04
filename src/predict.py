"""Reusable single-image inference."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.model import build_model
from src.preprocessing import build_eval_transform
from src.utils import get_device

log = logging.getLogger("idd.predict")

REQUIRED_CKPT_KEYS = (
    "state_dict", "architecture", "num_classes", "class_names", "input_size",
)
MAX_IMAGE_SIDE = 8192


class DefectPredictor:
    def __init__(self, checkpoint_path: Union[str, Path], device: torch.device | None = None) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if checkpoint_path.stat().st_size == 0:
            raise ValueError(f"Checkpoint is empty: {checkpoint_path}")

        self.device = device or get_device()
        log.info("Loading checkpoint %s on %s", checkpoint_path, self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")

        self.class_names = list(ckpt["class_names"])
        if len(self.class_names) != int(ckpt["num_classes"]):
            raise ValueError("class_names/num_classes mismatch")

        self.threshold = min(max(float(ckpt.get("threshold", 0.5)), 0.0), 1.0)
        self.input_size = (int(ckpt["input_size"][0]), int(ckpt["input_size"][1]))
        self.architecture = str(ckpt["architecture"])

        self.model = build_model(
            self.architecture, int(ckpt["num_classes"]), pretrained=False
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = build_eval_transform(self.input_size)
        self._defective_idx = self.class_names.index("Defective") \
            if "Defective" in self.class_names else len(self.class_names) - 1
        log.info("Model ready | classes=%s | threshold=%.3f | input=%s",
                 self.class_names, self.threshold, self.input_size)

    @staticmethod
    def _validate_image(image: Image.Image) -> None:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")
        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("Image has zero size.")
        if w > MAX_IMAGE_SIDE or h > MAX_IMAGE_SIDE:
            raise ValueError(f"Image too large: {w}x{h}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        self._validate_image(image)
        rgb = image.convert("RGB")

        t0 = time.perf_counter()
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs_t = torch.softmax(logits, dim=1)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probs = [min(max(float(p), 0.0), 1.0) for p in probs_t.cpu().tolist()]
        s = sum(probs)
        probs = [p / s for p in probs] if s > 0 else [1.0 / len(probs)] * len(probs)

        prob_defective = probs[self._defective_idx]
        pred_idx = self._defective_idx if prob_defective >= self.threshold else (
            0 if self._defective_idx != 0 else 1
        )

        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(probs))},
            "threshold": float(self.threshold),
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def info(self) -> dict:
        return {
            "architecture": self.architecture,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_size": list(self.input_size),
            "threshold": self.threshold,
            "device": str(self.device),
        }