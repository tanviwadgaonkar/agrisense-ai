"""End-to-end tests for the AgriSense pipeline.

Run from repo root:  python -m pytest -q
(assumes data + model exist: python data/make_dataset.py && python -m src.train)
"""
import pandas as pd
import pytest

from src.preprocess import FEATURES, MODELS_DIR, load_dataset, split, split_and_scale
from src.predict import load_artifacts, recommend
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
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()) == set(y_test.unique())
    # scaler is fit on training rows only
    assert abs(X_train.mean()).max() < 1e-9


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
    # Evaluate the *saved* model + scaler, exactly what the app serves.
    _, X_test, _, y_test = split(load_dataset())
    model, scaler = load_artifacts()
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
    assert model.score(X_test_s, y_test) > 0.85


def test_splitting_never_writes_artifacts():
    # split_and_scale used to overwrite models/scaler.joblib as a side effect,
    # so evaluating on a changed CSV silently desynced model and scaler.
    before = (MODELS_DIR / "scaler.joblib").stat().st_mtime_ns
    split_and_scale(load_dataset())
    assert (MODELS_DIR / "scaler.joblib").stat().st_mtime_ns == before


@pytest.mark.parametrize("field, value", [
    ("ph", 99), ("N", -500), ("humidity", float("nan")), ("rainfall", float("inf")), ("K", "abc"),
])
def test_recommend_rejects_bad_input(field, value):
    with pytest.raises(ValueError, match=field):
        recommend({**SAMPLE, field: value})


def test_recommend_rejects_missing_field():
    sample = dict(SAMPLE)
    del sample["ph"]
    with pytest.raises(ValueError, match="ph"):
        recommend(sample)


def test_flask_routes():
    from app.app import app
    client = app.test_client()
    assert client.get("/").status_code == 200
    resp = client.post("/", data={k: str(v) for k, v in SAMPLE.items()})
    assert resp.status_code == 200
    assert b"%" in resp.data  # a probability rendered


def test_flask_form_shows_validation_error():
    from app.app import app
    resp = app.test_client().post("/", data={**{k: str(v) for k, v in SAMPLE.items()}, "ph": "nan"})
    assert resp.status_code == 200
    assert b"ph must be a finite number" in resp.data


def test_json_api():
    from app.app import app
    client = app.test_client()
    ok = client.post("/api/recommend", json=SAMPLE)
    assert ok.status_code == 200
    assert len(ok.get_json()["recommendations"]) == 3
    assert client.post("/api/recommend", json={**SAMPLE, "ph": 42}).status_code == 422
    assert client.post("/api/recommend", data="not json", content_type="application/json").status_code == 400
    assert client.get("/health").get_json() == {"status": "ok"}
