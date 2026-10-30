
import importlib

flask_module = importlib.import_module("app.app")
import pytest

@pytest.fixture
def client():
    flask_module.app.config["TESTING"] = True
    return flask_module.app.test_client()
def fake_load_artifacts():
    raise FileNotFoundError("model.joblib missing")

def test_ready_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(flask_module,'load_artifacts',fake_load_artifacts)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.get_json()["status"] == "not ready"
def fake_successful_load():
    pass
    
def test_ready_when_artifacts_available(client, monkeypatch):
    monkeypatch.setattr(flask_module,'load_artifacts',fake_successful_load)
    response=client.get('/ready')
    assert response.status_code==200
    assert response.get_json()['status']=='ready'
