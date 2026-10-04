"""Regenerate split CSVs locally with the same seed as training."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
VAL_FRAC = 0.15

DATASET = Path(r"D:\dataset")
OUT = Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)


def discover(split_dir: Path):
    rows = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if "_GT" in p.stem:
            continue
        mask = p.with_name(f"{p.stem}_GT.png")
        label = 0
        if mask.exists():
            m = np.asarray(Image.open(mask).convert("L"))
            label = 1 if (m > 0).any() else 0
        rows.append({"path": str(p), "label": label, "mask": str(mask) if mask.exists() else ""})
    return rows


train_pool = discover(DATASET / "train")
test_pool  = discover(DATASET / "test")

by_cls = {0: [], 1: []}
for r in train_pool:
    by_cls[r["label"]].append(r)

rng = random.Random(SEED)
train, val = [], []
for cls, items in by_cls.items():
    rng.shuffle(items)
    n_val = max(1, int(len(items) * VAL_FRAC))
    val   += items[:n_val]
    train += items[n_val:]
rng.shuffle(train)
rng.shuffle(val)

pd.DataFrame(train).to_csv(OUT / "split_train.csv", index=False)
pd.DataFrame(val).to_csv(OUT / "split_val.csv", index=False)
pd.DataFrame(test_pool).to_csv(OUT / "split_test.csv", index=False)

print(f"train={len(train)}  val={len(val)}  test={len(test_pool)}")
print(f"train defective={sum(r['label'] for r in train)}")
print(f"val   defective={sum(r['label'] for r in val)}")
print(f"test  defective={sum(r['label'] for r in test_pool)}")
print(f"written to {OUT.resolve()}")
