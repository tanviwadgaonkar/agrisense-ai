"""Predict crops for one soil sample (shared by the CLI and the Flask app).

CLI:  python -m src.predict --N 78 --P 42 --K 40 --temperature 24 \
        --humidity 82 --ph 6.4 --rainfall 220 --soil_moisture 68
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocess import FEATURES
from src.soil_rules import soil_report
from functools import lru_cache

MODELS_DIR = Path("models")

@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return model, scaler


def recommend(sample: dict, top_k: int = 3) -> dict:
    """sample: dict with the 8 FEATURES. Returns soil report + top-k crops."""
    model, scaler = load_artifacts()
    X = pd.DataFrame([[sample[f] for f in FEATURES]], columns=FEATURES)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)

    proba = model.predict_proba(X_scaled)[0]
    order = np.argsort(proba)[::-1][:top_k]
    crops = [{"crop": model.classes_[i], "probability": round(float(proba[i]), 3)}
             for i in order]
    return {"soil": soil_report(sample), "recommendations": crops}


def main():
    parser = argparse.ArgumentParser(description="AgriSense crop recommendation")
    for f in FEATURES:
        parser.add_argument(f"--{f}", type=float, required=True)
    args = parser.parse_args()

    result = recommend(vars(args))
    print("Soil condition:", result["soil"]["overall"])
    for issue in result["soil"]["issues"]:
        print("  -", issue)
    print("Top crops:")
    for rec in result["recommendations"]:
        print(f"  {rec['crop']:14s} {rec['probability']:.1%}")


if __name__ == "__main__":
    main()
