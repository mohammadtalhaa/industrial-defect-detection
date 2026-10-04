# Industrial Defect Detection — Computer Vision Technical Assessment

<img width="2023" height="1608" alt="image" src="https://github.com/user-attachments/assets/6545dda1-8a28-48ff-9e6b-5dcd8ab04726" />


Binary image classifier that decides whether a manufactured surface is **Normal** or **Defective**, exposed through a FastAPI inference service and packaged for Docker.

> **Dataset disclosure.** The company intended to provide a proprietary dataset of production-line images. That dataset was not received before the submission deadline. After checking with the recruiter, this project uses **KolektorSDD2** (Kolektor Surface-Defect Dataset 2, ViCoS Lab, University of Ljubljana) as the development dataset. Source: https://www.vicos.si/resources/kolektorsdd2/. Licence: CC BY-NC-SA 4.0 — **non-commercial research use only**. The codebase is dataset-agnostic: point `paths.data_raw` at the company's dataset and rerun `scripts/make_splits.py` + training.

## 1. Overview

- **Task**: binary classification (Normal / Defective) on production-line surface images.
- **Model**: EfficientNet-B0, ImageNet-pretrained, fine-tuned end-to-end after a short frozen warm-up.
- **Imbalance**: inverse-frequency class weights in CrossEntropyLoss; validation F1-driven early stopping.
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, threshold sweep, FP/FN error analysis.
- **Serving**: FastAPI (`/predict`, `/health`, `/model/info`), model loaded once at startup, structured logging, proper HTTP codes.
- **Packaging**: Dockerfile + docker-compose with healthcheck.

## 2. Problem statement

Manual visual inspection on a production line is slow and inconsistent. This system ingests a product image and returns a class plus confidence, so operators only review flagged parts. **Recall on the Defective class is prioritised** because missed defects ship to customers, whereas false alarms only cost operator time.

## 3. Dataset

**KolektorSDD2** — 3,335 images, ~230×630 px, each with a pixel-level mask. Official statistics confirm: 3,335 total images, 356 defective, 2,979 normal, with 246 defective in the official training split and 110 in the test split.

| Split | Normal | Defective | Total |
|---|---|---|---|
| Official train | 2,085 | 246 | 2,331 |
| Official test  |   894 | 110 | 1,004 |
| **Total**      | 2,979 | 356 | 3,335 |

- Labels are derived from masks: an image is Defective if its `_GT.png` mask has any non-zero pixel.
- Official test split is **never** used for training, augmentation, hyperparameter or threshold selection.
- Training pool split 85/15 (stratified) into train/val with seed 42.

## 4. Preprocessing & augmentation

- **Aspect-preserving resize + pad** to 256×704 (source aspect ~230×630). Naive square resize would distort surface texture and shrink small defects.
- ImageNet mean/std normalization (matches pretrained weights).
- **Training augmentation**: ±5° rotation, small translation, mild brightness/contrast jitter. No aggressive transforms that could erase small defects.
- **Validation/test**: deterministic only.

## 5. Class imbalance

Defective ratio in the training split is ~10.6% (246 / 2,331). Handled with inverse-frequency weights:

```python
w_normal    = n / (2 * n_normal)
w_defective = n / (2 * n_defective)
```

Weighted CrossEntropy pushes the model to recall Defective images instead of defaulting to Normal. `WeightedRandomSampler` is implemented as an alternative in `src/train.py`.

## 6. Model

**EfficientNet-B0** (torchvision, ImageNet weights).

- Classification head replaced with a 2-way Linear.
- 2 epochs frozen backbone (warm-up) → full fine-tune at `lr/5`.
- Optimizer: AdamW (lr 1e-4, wd 1e-4), ReduceLROnPlateau, early stopping patience 5.
- Best checkpoint saved at **epoch 17**.

**Why EfficientNet-B0:**
- Strong ImageNet features, competitive on small datasets.
- ~5.3M params — runs CPU inference in ~80–230 ms per image.
- Well-supported by torchvision, easy to deploy.

## 7. Evaluation

All numbers below are from the **untouched official test split** (1,004 images: 110 Defective, 894 Normal), at a threshold locked from validation (0.50).

| Metric | Value |
|---|---|
| Accuracy | **0.9631** |
| Precision | **0.8411** |
| Recall (Defective) | **0.8182** |
| F1-score | **0.8295** |
| ROC-AUC | **0.9313** |
| PR-AUC | **0.8599** |
| Threshold | 0.50 |

**Confusion matrix**

|  | Predicted Normal | Predicted Defective |
|---|---|---|
| **Actual Normal** | 877 (TN) | 17 (FP) |
| **Actual Defective** | 20 (FN) | 90 (TP) |

**Operational interpretation**
- 18.2% of defective products were missed (20 / 110).
- 1.9% of normal products were false alarms (17 / 894).

Plots and CSVs: `artifacts/plots/confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `artifacts/metrics/metrics.json`, `artifacts/metrics/classification_report.json`, `artifacts/metrics/threshold_analysis.json`.

## 8. Error analysis

Every FP and FN image is copied to:
- `artifacts/error_analysis/false_positives/` — 17 images
- `artifacts/error_analysis/false_negatives/` — 20 images

Plus a per-image predictions CSV at `artifacts/error_analysis/predictions.csv` with `path, label, pred, prob_defective`.

**Observations**
- False negatives cluster on **subtle surface blemishes** (small spots, faint scratches) with low contrast — the model's feature map at 256×704 loses fine detail below ~2–3 pixels.
- False positives concentrate on **normal parts with strong edge texture** (scratches that look like defects but are within tolerance), producing high Defective scores.
- Business impact: FNs are the dangerous case. In deployment, the operating threshold should be tuned **downward** (e.g. 0.35) to push recall above 0.9 at the cost of more false alarms — cost of a false alarm ≪ cost of a shipped defect.

**Possible improvements**
- Multi-scale / higher input resolution (512×1408) to preserve fine defects.
- U-Net segmentation head trained from the same masks to localise and to reduce FN.
- Test-time augmentation + probability averaging.
- Hard-example mining using the FN list.

## 9. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Liveness + model-loaded flag |
| GET | `/model/info` | Architecture, classes, threshold |
| POST | `/predict` | Multipart image → prediction |

**Example**

```bash
curl -F "file=@sample.jpg" http://localhost:8000/predict
```

```json
{
  "filename": "sample.jpg",
  "predicted_class": "Defective",
  "confidence": 0.94,
  "probabilities": { "Normal": 0.06, "Defective": 0.94 },
  "threshold": 0.50,
  "inference_time_ms": 92.4
}
```

Confidence is a softmax score, not a guarantee.

**Production features**
- Model loaded once at startup (not per request).
- Input validation: extension whitelist, 10 MB cap, PIL decode check.
- Proper HTTP codes: 400 (bad input), 413 (too large), 415 (wrong type), 500 (inference), 503 (model unavailable).
- Structured logging per request with file, class, confidence, latency.
- CORS enabled.

## 10. Setup

```bash
git clone <your-repo> industrial-defect-detection
cd industrial-defect-detection
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-api.txt
```

Place the dataset at `D:\dataset\` (or `data/raw/dataset/`) with layout:

```
dataset/
├── train/   # <id>.jpg + <id>_GT.png
└── test/    # <id>.jpg + <id>_GT.png
```

Update `configs/config.yaml` → `paths.data_raw` to point at it.

## 11. Reproduce training

Training was performed on Google Colab (T4 GPU). The pipeline:

1. Read `<id>.jpg` + `<id>_GT.png` pairs from the official train folder.
2. Derive image-level labels from masks.
3. Stratified 85/15 train/val split of the official train pool (seed 42).
4. EfficientNet-B0, weighted CrossEntropy, AdamW, 2 frozen epochs then full fine-tune, early stopping.
5. Best checkpoint saved → `models/best_model.pt` (state dict + class names + input size + threshold).

To reproduce:

```bash
python scripts/make_splits.py                     # regenerate split CSVs
python -m src.train --config configs/config.yaml  # train
python -m src.evaluate --config configs/config.yaml  # threshold + test evaluation
```

## 12. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

## 13. Docker

```bash
docker compose up --build
curl http://localhost:8000/health
curl -F "file=@sample.jpg" http://localhost:8000/predict
```

The compose file mounts `./models` read-only so `best_model.pt` is picked up automatically.

## 14. Tests

```bash
pytest -q
```

Covers: model build, preprocessing shape, health endpoint, invalid extension, corrupt image, prediction schema. CI runs on GitHub Actions on every push.

## 15. Architecture

```
Dataset (KSDD2)
   → discover_samples()        (labels from GT masks)
   → split_train_val()         (val carved from official train, seed 42)
   → ResizeKeepAspectAndPad(256×704) + ImageNet normalization
   → EfficientNet-B0 (ImageNet pretrained)
   → weighted CrossEntropy + AdamW + ReduceLROnPlateau + early stopping
   → best_model.pt  (state_dict + class_names + input_size + threshold)
   → FastAPI /predict  →  JSON {class, confidence, probabilities, ms}
```

## 16. Known limitations

- **Image-level classification only** — no defect localisation. Cannot tell operators *where* the defect is.
- Trained on KolektorSDD2, not the company's actual production data — **domain shift is likely** when deployed.
- Non-commercial licence (CC BY-NC-SA 4.0) on the training data.
- 18.2% FN rate on the test split; threshold should be re-tuned on the target line.
- Input resolution 256×704 loses defects smaller than ~2–3 px.
- No drift monitoring or active-learning loop yet.

## 17. Future improvements

- Segmentation head (U-Net) using the same masks → defect localisation + recall boost.
- ONNX export + OpenVINO/TensorRT for lower latency.
- Test-time augmentation.
- Temperature scaling for calibrated confidence.
- Model registry + drift monitoring.
- Active learning on operator-flagged samples.

## 18. Citation

> Tabernik, D., Šela, S., Skvarč, J., Skočaj, D. (2020). *Segmentation-based deep-learning approach for surface-defect detection.* Journal of Intelligent Manufacturing, 31(3), 759–776. https://doi.org/10.1007/s10845-019-01476-x

Dataset page: https://www.vicos.si/resources/kolektorsdd2/
