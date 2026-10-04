"""Central configuration loader. All scripts read from configs/config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    seed: int
    raw: dict
    paths: dict
    data: dict
    augmentation: dict
    training: dict
    inference: dict
    _root: Path = field(default_factory=lambda: Path.cwd())

    # ---- convenience accessors ----
    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.data["image_height"]), int(self.data["image_width"])

    @property
    def class_names(self) -> List[str]:
        return list(self.data["class_names"])

    @property
    def checkpoint_path(self) -> Path:
        return self._root / self.paths["checkpoint"]

    @property
    def models_dir(self) -> Path:
        return self._root / self.paths["models_dir"]

    @property
    def artifacts_dir(self) -> Path:
        return self._root / self.paths["artifacts_dir"]

    def ensure_dirs(self) -> None:
        for p in (
            self.models_dir,
            self.artifacts_dir,
            self.artifacts_dir / "metrics",
            self.artifacts_dir / "plots",
            self.artifacts_dir / "error_analysis" / "false_positives",
            self.artifacts_dir / "error_analysis" / "false_negatives",
        ):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike = "configs/config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Run from the repository root."
        )
    with path.open() as f:
        raw = yaml.safe_load(f)
    root = path.resolve().parent.parent
    return Config(
        seed=raw.get("seed", 42),
        raw=raw,
        paths=raw["paths"],
        data=raw["data"],
        augmentation=raw["augmentation"],
        training=raw["training"],
        inference=raw["inference"],
        _root=root,
    )