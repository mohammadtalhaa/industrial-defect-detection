"""Deterministic preprocessing + ImageNet normalization.
Aspect-preserving resize + pad so defect geometry is not distorted."""
from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeKeepAspectAndPad:
    """Resize longest side to target, pad to exact (H, W)."""

    def __init__(self, size: Tuple[int, int], fill: int = 0):
        self.h, self.w = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.w / w, self.h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.w, self.h), (self.fill,) * 3)
        canvas.paste(img, ((self.w - new_w) // 2, (self.h - new_h) // 2))
        return canvas


def build_eval_transform(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeKeepAspectAndPad(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalize a CHW uint8/float tensor with ImageNet statistics."""
    if t.dtype != torch.float32:
        t = t.float() / 255.0
    return TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)