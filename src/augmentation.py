"""Conservative augmentation for training only."""
from __future__ import annotations

from typing import Tuple

from torchvision import transforms

from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD, ResizeKeepAspectAndPad


def build_train_transform(size: Tuple[int, int], aug_cfg: dict) -> transforms.Compose:
    ops = [
        ResizeKeepAspectAndPad(size),
        transforms.RandomApply(
            [transforms.RandomRotation(aug_cfg.get("rotation_deg", 5))], p=0.5
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(aug_cfg.get("translate_frac", 0.02),) * 2,
        ),
        transforms.ColorJitter(
            brightness=aug_cfg.get("brightness", 0.1),
            contrast=aug_cfg.get("contrast", 0.1),
        ),
    ]
    if aug_cfg.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)