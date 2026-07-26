"""End-to-end tests for the AgriSense pipeline.

Run from repo root:  python -m pytest -q
(assumes data + model exist: python data/make_dataset.py && python -m src.train)
"""
import pandas as pd
import pytest

from src.preprocess import FEATURES, load_dataset, split_and_scale
from src.predict import recommend
from src.soil_rules import classify_ph, soil_report

SAMPLE = {"N": 78, "P": 42, "K": 40, "temperature": 24, "humidity": 82,
          "ph": 6.4, "rainfall": 220, "soil_moisture": 68}


def test_dataset_loads_clean():
    df = load_dataset()
    assert len(df) > 1000
    assert not df[FEATURES].isna().any().any()
    assert df["label"].nunique() >= 10


def test_split_shapes_and_stratify():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_and_scale(df)
    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()) == set(y_test.unique())


def test_soil_rules():
    assert classify_ph(4.9) == "strongly acidic"
    assert classify_ph(7.0) == "neutral"
    report = soil_report({**SAMPLE, "N": 10})
    assert report["nitrogen"] == "low"
    assert any("N is low" in issue for issue in report["issues"])


def test_recommend_returns_valid_crops():
    result = recommend(SAMPLE)
    recs = result["recommendations"]
    assert len(recs) == 3
    assert recs[0]["probability"] >= recs[1]["probability"] >= recs[2]["probability"]
    assert 0 <= recs[0]["probability"] <= 1
    # rice and jute have overlapping agronomy; either is a valid top pick here
    assert recs[0]["crop"] in {"rice", "jute"}


def test_recommend_distinct_profile():
    # grapes have a very distinctive P/K signature -> deterministic top-1
    grapes = {"N": 20, "P": 130, "K": 200, "temperature": 22, "humidity": 81,
              "ph": 6.0, "rainfall": 70, "soil_moisture": 35}
    result = recommend(grapes)
    assert result["recommendations"][0]["crop"] == "grapes"


def test_model_accuracy_floor():
    import joblib
    df = load_dataset()
    _, X_test, _, y_test = split_and_scale(df)
    model = joblib.load("models/model.joblib")
    assert model.score(X_test, y_test) > 0.85


def test_flask_routes():
    from app.app import app
    client = app.test_client()
    assert client.get("/").status_code == 200
    resp = client.post("/", data={k: str(v) for k, v in SAMPLE.items()})
    assert resp.status_code == 200
    assert b"%" in resp.data  # a probability rendered
