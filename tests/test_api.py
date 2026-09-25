from fastapi.testclient import TestClient

from app import app, feature_names

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["api_version"]


def test_docs_available():
    assert client.get("/docs").status_code == 200


def test_sample_shape():
    response = client.get("/api/v1/sample")
    assert response.status_code == 200
    payload = response.json()
    assert "features" in payload
    assert len(payload["features"]) == len(feature_names)


def test_model_info():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ElasticNet Regression"
    assert body["feature_count"] == 81


def test_invalid_prediction_rejected_without_training():
    response = client.post("/api/v1/predict", json={"features": {"number_of_elements": 2.0}})
    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"]


def test_schema():
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    assert response.json()["feature_count"] == 81
