"""Training entry point. Designed for Colab GPU or CPU fallback."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import build_train_transform
from src.config import load_config
from src.data_loader import (
    DefectDataset,
    discover_samples,
    samples_to_dataframe,
    split_train_val,
)
from src.model import build_model, count_trainable, freeze_backbone, unfreeze_all
from src.preprocessing import build_eval_transform
from src.utils import get_device, get_logger, save_json, set_seed


def _compute_class_weights(samples, device):
    labels = [s.label for s in samples]
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    # inverse-frequency weights, Defective up-weighted
    w = torch.tensor([n / (2 * max(n_neg, 1)), n / (2 * max(n_pos, 1))], dtype=torch.float)
    return w.to(device)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds.extend(logits.argmax(1).detach().cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return avg_loss, f1


def train(cfg_path: str = "configs/config.yaml", use_official_split: bool = True):
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    log = get_logger()
    device = get_device()
    log.info(f"Device: {device}")

    root = cfg._root / cfg.paths["data_raw"]
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir}. "
            "See scripts/prepare_dataset.py for expected layout."
        )

    train_samples = discover_samples(train_dir)
    test_samples = discover_samples(test_dir)
    log.info(f"Train pool: {len(train_samples)} | Test: {len(test_samples)}")
    tr, va = split_train_val(train_samples, cfg.data["val_split"], cfg.seed)
    log.info(f"Split -> train={len(tr)} val={len(va)}")
    log.info(
        f"Defective ratio -> train={sum(s.label for s in tr)/len(tr):.3f} "
        f"val={sum(s.label for s in va)/len(va):.3f} "
        f"test={sum(s.label for s in test_samples)/len(test_samples):.3f}"
    )

    # Save split manifests (reproducibility)
    samples_to_dataframe(tr).to_csv(cfg.artifacts_dir / "split_train.csv", index=False)
    samples_to_dataframe(va).to_csv(cfg.artifacts_dir / "split_val.csv", index=False)
    samples_to_dataframe(test_samples).to_csv(cfg.artifacts_dir / "split_test.csv", index=False)

    train_tf = build_train_transform(cfg.image_size, cfg.augmentation)
    eval_tf = build_eval_transform(cfg.image_size)

    train_loader = DataLoader(
        DefectDataset(tr, train_tf),
        batch_size=cfg.training["batch_size"],
        shuffle=True,
        num_workers=cfg.data["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        DefectDataset(va, eval_tf),
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        num_workers=cfg.data["num_workers"],
    )

    model = build_model(
        cfg.training["architecture"],
        num_classes=cfg.data["num_classes"],
        pretrained=cfg.training["pretrained"],
    ).to(device)

    freeze_backbone(model, cfg.training["architecture"])
    log.info(f"Frozen backbone. Trainable params: {count_trainable(model):,}")

    if cfg.training["use_class_weights"]:
        weights = _compute_class_weights(tr, device)
        log.info(f"Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training["lr"],
        weight_decay=cfg.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1, epochs_no_improve = -1.0, 0
    freeze_epochs = cfg.training["freeze_backbone_epochs"]

    for epoch in range(1, cfg.training["epochs"] + 1):
        if epoch == freeze_epochs + 1:
            unfreeze_all(model)
            log.info(f"Epoch {epoch}: unfroze backbone. Trainable: {count_trainable(model):,}")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.training["lr"] / 5, weight_decay=cfg.training["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        tr_loss, tr_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_f1 = _run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)
        log.info(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
            f"| val_loss={va_loss:.4f} val_f1={va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": cfg.training["architecture"],
                    "num_classes": cfg.data["num_classes"],
                    "class_names": cfg.class_names,
                    "input_size": list(cfg.image_size),
                    "threshold": cfg.inference["threshold"],
                    "val_f1": best_f1,
                    "epoch": epoch,
                },
                cfg.checkpoint_path,
            )
            log.info(f"  ↳ saved best checkpoint (val_f1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training["patience"]:
                log.info("Early stopping triggered.")
                break

    save_json(history, cfg.artifacts_dir / "metrics" / "training_history.json")
    log.info(f"Training done. Best val F1 = {best_f1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    train(args.config)