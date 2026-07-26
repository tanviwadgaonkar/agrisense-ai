"""Load, clean and split the soil dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture"]
TARGET = "label"
DATA_PATH = Path("data/soil_data.csv")
MODELS_DIR = Path("models")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    # basic cleaning: drop duplicates / nulls, keep physically sane rows
    df = df.drop_duplicates().dropna(subset=FEATURES + [TARGET])
    df = df[(df["ph"].between(3, 10)) & (df["humidity"].between(0, 100))]
    df = df[(df["soil_moisture"].between(0, 100)) & (df["rainfall"] >= 0)]
    return df.reset_index(drop=True)


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    return X_train_s, X_test_s, y_train.reset_index(drop=True), y_test.reset_index(drop=True)
