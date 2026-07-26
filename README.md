# AgriSense AI — Soil Classification & Crop Recommendation

End-to-end machine-learning application that **classifies soil condition**
(rule-based agronomy report: pH class, N/P/K levels, moisture state) and
**recommends the best crops** (ML classifier) from soil-test values —
N, P, K, pH, soil moisture — plus environmental attributes (temperature,
humidity, rainfall).

> Related publication: *soil classification and crop recommendation*,
> accepted for presentation at **ICAIMEST-2026**.

## How it works

```
soil test values ──► preprocessing (clean, validate, scale)
                      ├─► rule-based soil report (pH / N / P / K / moisture)
                      └─► ML model (best of LR / RF / GB) ──► top-3 crops + probabilities
                                        │
                                Flask web app UI
```

- `data/make_dataset.py` — builds the sample dataset (2,160 rows, 12 crops, 8 features)
- `src/preprocess.py` — cleaning, validation, stratified split, feature scaling
- `src/train.py` — trains & compares 3 models with 5-fold CV, saves the best + metrics + plots
- `src/soil_rules.py` — agronomy rules for soil condition classification
- `src/predict.py` — shared inference (CLI + app), top-k crops with probabilities
- `app/app.py` — Flask interface for user-facing soil analysis

## Results (test set, sample dataset)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | 5-fold CV |
|---|---|---|---|---|---|
| Logistic Regression | 91.20% | 0.913 | 0.912 | 0.912 | 0.904 |
| **Random Forest** ✅ | **95.14%** | **0.952** | **0.951** | **0.952** | **0.938** |
| Gradient Boosting | 93.98% | 0.940 | 0.940 | 0.940 | 0.927 |

![confusion matrix](reports/confusion_matrix.png)
![feature importance](reports/feature_importance.png)

> **Note on data:** the included dataset is a synthetic sample with the same
> schema as the study dataset (generated in `data/make_dataset.py` from
> per-crop agronomic ranges, with label noise). To reproduce the paper's
> numbers, place the study CSV at `data/soil_data.csv` and re-run training —
> all metrics, plots and the README table regenerate from `models/metrics.json`.

## Run it

```bash
pip install -r requirements.txt

python data/make_dataset.py     # or drop in your own soil_data.csv
python -m src.train             # trains, compares, saves best model
python -m src.evaluate          # full classification report
python -m pytest -q             # 7 tests

python -m app.app               # Flask UI at http://localhost:5000
```

CLI prediction:

```bash
python -m src.predict --N 78 --P 42 --K 40 --temperature 24 \
  --humidity 82 --ph 6.4 --rainfall 220 --soil_moisture 68
```

## Stack

Python · scikit-learn · pandas · NumPy · Flask · Matplotlib · pytest
