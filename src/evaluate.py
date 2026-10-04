"""Evaluate the trained checkpoint on the untouched official test split.
Selects the decision threshold on validation, then locks it for test."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data_loader import DefectDataset, Sample, discover_samples, split_train_val
from src.model import build_model
from src.preprocessing import build_eval_transform
from src.utils import get_device, get_logger, load_json, save_json, set_seed


def _load_samples_from_csv(path: Path) -> list[Sample]:
    import pandas as pd

    df = pd.read_csv(path)
    return [Sample(path=Path(r["path"]), label=int(r["label"])) for _, r in df.iterrows()]


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs, labels, paths = [], [], []
    for x, y, p in tqdm(loader, leave=False):
        x = x.to(device)
        logits = model(x)
        prob_pos = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(prob_pos.cpu().tolist())
        labels.extend(y.tolist())
        paths.extend(list(p))
    return np.array(probs), np.array(labels), paths


def sweep_threshold(y_true, y_prob):
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return rows


def evaluate(cfg_path: str = "configs/config.yaml"):
    cfg = load_config(cfg_path)
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    log = get_logger()
    device = get_device()

    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    log.info(f"Loaded checkpoint from {cfg.checkpoint_path} (epoch {ckpt.get('epoch')})")

    model = build_model(
        ckpt["architecture"], num_classes=ckpt["num_classes"], pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    eval_tf = build_eval_transform(tuple(ckpt["input_size"]))

    # --- validation set for threshold ---
    val_samples = _load_samples_from_csv(cfg.artifacts_dir / "split_val.csv")
    val_loader = DataLoader(DefectDataset(val_samples, eval_tf), batch_size=16, shuffle=False)
    val_probs, val_labels, _ = _predict(model, val_loader, device)
    sweep = sweep_threshold(val_labels, val_probs)
    best = max(sweep, key=lambda r: r["f1"])
    threshold = best["threshold"]
    log.info(f"Threshold selected on VALIDATION: {threshold:.2f} (val F1={best['f1']:.4f})")
    save_json({"val_sweep": sweep, "selected_threshold": threshold},
              cfg.artifacts_dir / "metrics" / "threshold_analysis.json")

    # --- locked evaluation on official test ---
    test_samples = _load_samples_from_csv(cfg.artifacts_dir / "split_test.csv")
    test_loader = DataLoader(DefectDataset(test_samples, eval_tf), batch_size=16, shuffle=False)
    probs, labels, paths = _predict(model, test_loader, device)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    log.info(f"TEST metrics: {metrics}")
    save_json(metrics, cfg.artifacts_dir / "metrics" / "metrics.json")
    save_json(
        classification_report(labels, preds, target_names=cfg.class_names, output_dict=True),
        cfg.artifacts_dir / "metrics" / "classification_report.json",
    )

    # --- plots ---
    plots = cfg.artifacts_dir / "plots"
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.class_names, yticklabels=cfg.class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(plots / "confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(plots / "roc_curve.png", dpi=150); plt.close()

    p, r, _ = precision_recall_curve(labels, probs)
    plt.figure(); plt.plot(r, p, label=f"AP={metrics['pr_auc']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots / "precision_recall_curve.png", dpi=150); plt.close()

    # --- error analysis ---
    fp_dir = cfg.artifacts_dir / "error_analysis" / "false_positives"
    fn_dir = cfg.artifacts_dir / "error_analysis" / "false_negatives"
    import shutil
    from PIL import Image

    rows = []
    for path, y, yhat, prob in zip(paths, labels, preds, probs):
        if yhat != y:
            dst = (fp_dir if (yhat == 1 and y == 0) else fn_dir) / Path(path).name
            shutil.copy(path, dst)
        rows.append({"path": path, "label": int(y), "pred": int(yhat), "prob_defective": float(prob)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(cfg.artifacts_dir / "error_analysis" / "predictions.csv", index=False)

    # Update checkpoint with the validated threshold so the API uses it
    ckpt["threshold"] = threshold
    torch.save(ckpt, cfg.checkpoint_path)
    log.info("Evaluation complete. Checkpoint threshold updated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    evaluate(args.config)