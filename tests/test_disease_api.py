from app.app import app
from io import BytesIO
def test_missing_image():
    client=app.test_client()
    request=client.post("/predict-disease")
    assert request.status_code==400
def test_invalid_image():
    client=app.test_client()
    fake_file=BytesIO(b"this is a fake image")
    request=client.post("/predict-disease",data={"image":(fake_file,"fake.jpg")},content_type="multipath/form-data")
    assert request.status_code==400