"""Train and compare crop-recommendation models, keep the best one.

Run:  python -m src.train
Outputs: models/model.joblib, models/scaler.joblib, models/metrics.json,
         reports/confusion_matrix.png, reports/feature_importance.png
"""
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import FEATURES, MODELS_DIR, ROOT, load_dataset, split_and_scale

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}
REPORTS = ROOT / "reports"


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_macro": round(precision_score(y_test, pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
    }


def plot_confusion(model, X_test, y_test, labels):
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted crop"); ax.set_ylabel("Actual crop")
    ax.set_title("Confusion matrix - best model")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_importance(model):
    if not hasattr(model, "feature_importances_"):
        return
    order = np.argsort(model.feature_importances_)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(FEATURES)[order], model.feature_importances_[order], color="#3a7d44")
    ax.set_title("Feature importance - best model")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)


def cross_validate(model, X_train_raw, y_train) -> np.ndarray:
    """5-fold CV with the scaler *inside* the pipeline, so each fold's scaler
    is fit on that fold's training rows only (no leakage from validation rows)."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(StandardScaler(), clone(model))
    return cross_val_score(pipe, X_train_raw, y_train, cv=folds, scoring="f1_macro")


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    X_train_raw = scaler.inverse_transform(X_train)

    results, fitted = {}, {}
    for name, model in MODELS.items():
        cv = cross_validate(model, X_train_raw, y_train)
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test) | {
            "cv_f1_macro_mean": round(cv.mean(), 4),
            "cv_f1_macro_std": round(cv.std(), 4),
        }
        fitted[name] = model
        print(f"{name:22s} cv_f1={cv.mean():.4f}±{cv.std():.4f}  "
              f"test_acc={results[name]['accuracy']:.4f}  test_f1={results[name]['f1_macro']:.4f}")

    # Choose on cross-validation (training data only). The test set is used
    # once, for reporting; choosing on it would make the reported score optimistic.
    best_name = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])
    best = fitted[best_name]
    print(f"\nbest model (by CV macro-F1): {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODELS_DIR / "model.joblib", compress=3)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    metrics = {"best_model": best_name, "selected_by": "cv_f1_macro_mean",
               "n_samples": len(df), "n_classes": int(df["label"].nunique()),
               "trained_with": {"python": platform.python_version(),
                                "scikit_learn": sklearn.__version__},
               "results": results}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion(best, X_test, y_test, sorted(df["label"].unique()))
    plot_importance(best)
    print("saved: models/model.joblib, models/metrics.json, reports/*.png")


if __name__ == "__main__":
    main()
