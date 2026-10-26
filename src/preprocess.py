"""Load, clean, validate and split the soil dataset."""
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture"]
TARGET = "label"
DATA_PATH = ROOT / "data" / "soil_data.csv"
MODELS_DIR = ROOT / "models"

# Physically plausible input ranges (inclusive). The same limits clean the
# training data and validate user input, so the model is never asked about
# values it could not have seen.
LIMITS = {
    "N": (0, 300),
    "P": (0, 300),
    "K": (0, 300),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (3, 10),
    "rainfall": (0, 3000),
    "soil_moisture": (0, 100),
}


def validate_sample(sample: dict) -> dict:
    """Return ``{feature: float}`` or raise ValueError naming the bad field."""
    clean = {}
    for f in FEATURES:
        if f not in sample:
            raise ValueError(f"missing field: {f}")
        try:
            value = float(sample[f])
        except (TypeError, ValueError):
            raise ValueError(f"{f} must be a number") from None
        if not math.isfinite(value):
            raise ValueError(f"{f} must be a finite number")
        lo, hi = LIMITS[f]
        if not lo <= value <= hi:
            raise ValueError(f"{f} must be between {lo} and {hi}")
        clean[f] = value
    return clean


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    # basic cleaning: drop duplicates / nulls, keep physically sane rows
    df = df.drop_duplicates().dropna(subset=FEATURES + [TARGET])
    in_range = pd.Series(True, index=df.index)
    for f, (lo, hi) in LIMITS.items():
        in_range &= df[f].between(lo, hi)
    return df[in_range].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split of the raw (unscaled) features."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split, then fit a scaler on the training rows only.

    Returns ``(X_train_s, X_test_s, y_train, y_test, scaler)``. Nothing is
    written to disk: saving the scaler is the training script's job, so
    evaluation and tests can never overwrite the artifact the model needs.
    """
    X_train, X_test, y_train, y_test = split(df, test_size, seed)
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
    return X_train_s, X_test_s, y_train, y_test, scaler
