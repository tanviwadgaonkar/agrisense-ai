"""Upload validation for /predict-disease (no disease model is trained yet)."""
from io import BytesIO

import pytest
from PIL import Image

from app.app import app


@pytest.fixture
def client():
    return app.test_client()


def _image(fmt: str) -> BytesIO:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "green").save(buf, format=fmt)
    buf.seek(0)
    return buf


def _post(client, file, name):
    # The original test sent "multipath/form-data" (typo), so no file was ever
    # uploaded and it passed via the "image is not found" branch.
    return client.post("/predict-disease", data={"image": (file, name)},
                       content_type="multipart/form-data")


def test_missing_image(client):
    resp = client.post("/predict-disease")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image is not found"


def test_invalid_image_bytes(client):
    resp = _post(client, BytesIO(b"this is a fake image"), "fake.jpg")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Image Invalid"


def test_unsupported_format(client):
    resp = _post(client, _image("GIF"), "leaf.gif")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Unsupported Image Format"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_valid_image_reports_model_unavailable(client, fmt):
    resp = _post(client, _image(fmt), f"leaf.{fmt.lower()}")
    assert resp.status_code == 501
    assert resp.get_json()["image"] == {"format": fmt, "width": 8, "height": 8}


def test_oversized_upload_is_413(client):
    resp = _post(client, BytesIO(b"0" * (5 * 1024 * 1024 + 1)), "huge.png")
    assert resp.status_code == 413
