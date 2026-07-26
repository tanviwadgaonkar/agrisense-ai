"""Print a full classification report for the saved model.

Run:  python -m src.evaluate
"""
import joblib
from sklearn.metrics import classification_report

from src.preprocess import load_dataset, split_and_scale


def main():
    df = load_dataset()
    _, X_test, _, y_test = split_and_scale(df)
    model = joblib.load("models/model.joblib")
    print(classification_report(y_test, model.predict(X_test)))


if __name__ == "__main__":
    main()
