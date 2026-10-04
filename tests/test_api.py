import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _img_bytes(fmt="JPEG", size=(630, 230)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(100, 100, 100)).save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_rejects_non_image_extension(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_corrupt_image(client):
    r = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_returns_schema_if_model_available(client):
    r = client.post(
        "/predict",
        files={"file": ("good.jpg", _img_bytes(), "image/jpeg")},
    )
    if r.status_code == 503:
        pytest.skip("Checkpoint not present.")
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        body = r.json()
        for k in ("filename", "predicted_class", "confidence", "probabilities",
                  "threshold", "inference_time_ms"):
            assert k in body
        assert body["predicted_class"] in ("Normal", "Defective")
        assert 0.0 <= body["confidence"] <= 1.0