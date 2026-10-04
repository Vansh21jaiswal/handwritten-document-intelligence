from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "tesseract" in response.json()["available_models"]

def test_predict_invalid_file_format():
    # Sending a plain text file instead of an image
    files = {"file": ("test.txt", b"this is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]
