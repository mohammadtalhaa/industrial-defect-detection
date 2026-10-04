"""FastAPI inference service."""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from api.dependencies import get_predictor
from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("idd.api")

MAX_UPLOAD_MB = 10
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model at startup...")
    try:
        get_predictor()
        log.info("Model ready.")
    except Exception as e:
        log.exception("Failed to load model at startup: %s", e)
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["info"])
def root():
    return {
        "service": "Industrial Defect Detection API",
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/predict"],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    try:
        get_predictor()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["info"])
def model_info():
    try:
        p = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")
    return {
        "architecture": type(p.model).__name__,
        "num_classes": len(p.class_names),
        "class_names": p.class_names,
        "input_size": list(p.input_size),
        "threshold": p.threshold,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB).")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {e}")

    t0 = time.perf_counter()
    try:
        result = get_predictor().predict(image)
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    log.info(
        "predict file=%s class=%s conf=%.3f total_ms=%.1f",
        file.filename, result["predicted_class"], result["confidence"],
        (time.perf_counter() - t0) * 1000.0,
    )
    return {"filename": file.filename, **result}