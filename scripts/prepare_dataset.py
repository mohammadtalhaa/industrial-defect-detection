"""Utility: verify the expected on-disk layout and print a dataset report."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data_loader import discover_samples


EXPECTED = """
Expected layout (KolektorSDD2):

  data/raw/kolektorsdd2/
  ├── train/
  │   ├── 10001.jpg
  │   ├── 10001_GT.png       # mask; all-zero = Normal, non-zero = Defective
  │   └── ...
  └── test/
      ├── 20001.jpg
      ├── 20001_GT.png
      └── ...

Download: https://www.vicos.si/resources/kolektorsdd2/
"""


def main(root: str):
    root = Path(root)
    if not root.exists():
        print(EXPECTED)
        raise SystemExit(f"Not found: {root}")

    for split in ("train", "test"):
        d = root / split
        if not d.exists():
            print(f"[!] Missing split dir: {d}")
            continue
        samples = discover_samples(d)
        counts = Counter(s.label for s in samples)
        print(f"{split}: total={len(samples)} normal={counts[0]} defective={counts[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kolektorsdd2")
    main(ap.parse_args().root)