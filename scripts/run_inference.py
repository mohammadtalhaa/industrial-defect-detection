"""CLI: run the trained checkpoint on a single image."""
from __future__ import annotations

import argparse
import json

from PIL import Image

from src.predict import DefectPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    predictor = DefectPredictor(args.checkpoint)
    result = predictor.predict(Image.open(args.image))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()