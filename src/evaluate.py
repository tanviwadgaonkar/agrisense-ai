"""Print a full classification report for the saved model and scaler.

Run:  python -m src.evaluate

Uses the *saved* artifacts on the held-out split — it evaluates exactly what
the app serves, and never refits or overwrites anything.
"""
from sklearn.metrics import classification_report

from src.predict import load_artifacts
from src.preprocess import FEATURES, load_dataset, split
import pandas as pd


def main():
    _, X_test, _, y_test = split(load_dataset())
    model, scaler = load_artifacts()
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
    print(classification_report(y_test, model.predict(X_test_s)))


if __name__ == "__main__":
    main()
