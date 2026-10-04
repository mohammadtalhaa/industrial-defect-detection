"""Dataset discovery + PyTorch Dataset for KolektorSDD2.

KSDD2 ships with images (`<id>.jpg`) and masks (`<id>_GT.png`).
An image is defective if its mask has any non-zero pixel.
Structure is auto-discovered so this also works if the company later
drops in their own dataset with the same naming convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: Path
    label: int  # 0 normal, 1 defective
    mask_path: Path | None = None


def _mask_is_defective(mask_path: Path) -> bool:
    if not mask_path.exists():
        return False
    m = np.asarray(Image.open(mask_path).convert("L"))
    return bool((m > 0).any())


def discover_samples(root: Path) -> List[Sample]:
    """Find all non-mask images under `root` and label them via their GT mask."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples: List[Sample] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in img_exts:
            continue
        if "_GT" in p.stem:  # skip mask files
            continue
        mask = p.with_name(f"{p.stem}_GT.png")
        label = 1 if _mask_is_defective(mask) else 0
        samples.append(Sample(path=p, label=label, mask_path=mask if mask.exists() else None))
    return samples


def split_train_val(
    samples: Sequence[Sample],
    val_frac: float,
    seed: int,
) -> tuple[List[Sample], List[Sample]]:
    import random

    rng = random.Random(seed)
    by_class: dict[int, List[Sample]] = {0: [], 1: []}
    for s in samples:
        by_class[s.label].append(s)
    train, val = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"path": str(s.path), "label": s.label, "mask": str(s.mask_path) if s.mask_path else ""} for s in samples]
    )


class DefectDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(s.label, dtype=torch.long), str(s.path)