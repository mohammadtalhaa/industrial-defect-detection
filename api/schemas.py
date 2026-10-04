from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    threshold: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes: int
    class_names: list[str]
    input_size: list[int]
    threshold: float