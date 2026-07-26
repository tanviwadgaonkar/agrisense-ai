"""Generate the sample soil/crop dataset used for the demo build.

Each crop has agronomically plausible ranges for soil nutrients (N, P, K),
pH, soil moisture and environmental attributes (temperature, humidity,
rainfall). Rows are sampled inside those ranges with noise, plus a small
amount of label noise so the classification task stays realistic.

NOTE: this is a synthetic stand-in with the same schema as the dataset used
in the ICAIMEST-2026 paper. To reproduce the paper's numbers, drop the real
CSV in data/soil_data.csv and re-run src/train.py.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# crop: (N, P, K, temperature C, humidity %, ph, rainfall mm, soil_moisture %)
CROP_PROFILES = {
    "rice":       ((60, 99),  (35, 60),  (35, 45),  (20, 27), (78, 86), (5.0, 7.0), (180, 300), (60, 80)),
    "maize":      ((60, 100), (35, 60),  (15, 25),  (18, 27), (55, 75), (5.5, 7.0), (60, 110),  (30, 50)),
    "wheat":      ((50, 90),  (30, 55),  (20, 35),  (12, 25), (50, 70), (6.0, 7.5), (45, 90),   (25, 45)),
    "chickpea":   ((20, 60),  (55, 80),  (75, 85),  (17, 21), (14, 22), (5.9, 8.0), (65, 95),   (15, 30)),
    "kidneybeans":((0, 40),   (55, 80),  (15, 25),  (15, 25), (18, 25), (5.5, 6.0), (60, 150),  (20, 35)),
    "banana":     ((80, 120), (70, 95),  (45, 55),  (25, 30), (75, 85), (5.5, 6.5), (90, 120),  (40, 60)),
    "mango":      ((0, 40),   (15, 35),  (25, 35),  (27, 35), (45, 55), (4.5, 7.0), (80, 110),  (20, 40)),
    "grapes":     ((0, 40),   (120, 145),(195, 205),(10, 32), (78, 84), (5.5, 6.5), (65, 75),   (25, 45)),
    "cotton":     ((100, 140),(35, 60),  (15, 25),  (22, 26), (75, 85), (5.8, 8.0), (60, 100),  (25, 45)),
    "sugarcane":  ((80, 120), (30, 60),  (40, 60),  (21, 35), (65, 80), (6.0, 8.0), (150, 250), (50, 70)),
    "coffee":     ((80, 120), (15, 40),  (25, 35),  (23, 28), (50, 70), (6.0, 7.0), (110, 180), (35, 55)),
    "jute":       ((60, 100), (35, 60),  (35, 45),  (23, 27), (70, 90), (6.0, 7.5), (150, 250), (55, 75)),
}

COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture"]
ROWS_PER_CROP = 180
LABEL_NOISE = 0.04


def _sample(lo, hi, n):
    """Sample uniformly inside a range, then jitter past the edges slightly."""
    base = RNG.uniform(lo, hi, n)
    jitter = RNG.normal(0, (hi - lo) * 0.06, n)
    return base + jitter


def build(rows_per_crop: int = ROWS_PER_CROP) -> pd.DataFrame:
    frames = []
    for crop, ranges in CROP_PROFILES.items():
        cols = {c: _sample(lo, hi, rows_per_crop) for c, (lo, hi) in zip(COLUMNS, ranges)}
        df = pd.DataFrame(cols)
        df["label"] = crop
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # clip to physical limits
    data["ph"] = data["ph"].clip(3.5, 9.5)
    data["humidity"] = data["humidity"].clip(10, 100)
    data["soil_moisture"] = data["soil_moisture"].clip(5, 95)
    for col in ("N", "P", "K", "rainfall"):
        data[col] = data[col].clip(lower=0)

    # small amount of label noise -> keeps accuracy honest
    n_noisy = int(len(data) * LABEL_NOISE)
    idx = RNG.choice(len(data), n_noisy, replace=False)
    labels = list(CROP_PROFILES)
    data.loc[idx, "label"] = [RNG.choice(labels) for _ in idx]

    return data.sample(frac=1, random_state=42).reset_index(drop=True).round(2)


if __name__ == "__main__":
    df = build()
    df.to_csv("data/soil_data.csv", index=False)
    print(f"wrote data/soil_data.csv  ->  {df.shape[0]} rows, {df['label'].nunique()} crops")
