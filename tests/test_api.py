import io

from fastapi.testclient import TestClient

from app import app, feature_names

client = TestClient(app)


def get_sample():
    response = client.get("/api/v1/sample")
    assert response.status_code == 200
    return response.json()


def test_homepage_and_docs_available():
    home = client.get("/")
    assert home.status_code == 200
    assert "Critical Temperature Predictor" in home.text
    assert client.get("/docs").status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["api_version"]
    assert body["model_version"]


def test_sample_shape():
    payload = get_sample()
    assert "features" in payload
    assert len(payload["features"]) == len(feature_names) == 81


def test_model_info():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "ElasticNet Regression"
    assert body["feature_count"] == 81
    assert body["validated_test_metrics"]["r2"] == 0.737


def test_invalid_prediction_rejected_without_training():
    response = client.post(
        "/api/v1/predict",
        json={"features": {"number_of_elements": 2.0}},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"]


def test_valid_prediction_returns_engineering_metadata():
    payload = get_sample()
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["predicted_critical_temp_k"], float)
    assert body["feature_count"] == 81
    assert body["model_version"] == "1.0.0"
    assert len(body["top_contributors"]) == 10
    assert body["response_time_ms"] >= 0
    assert body["request_id"]
    assert 0 <= body["ood_risk_score"] <= 100


def test_schema_has_group_descriptions_and_stats():
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] == 81

    feature = body["features"]["mean_atomic_mass"]
    assert feature["group"] == "Atomic Mass"
    assert feature["description"]
    assert "min" in feature
    assert "median" in feature
    assert "max" in feature


def test_top_features():
    response = client.get("/api/v1/top-features?limit=5")
    assert response.status_code == 200
    features = response.json()["features"]
    assert len(features) == 5
    assert all("coefficient" in item for item in features)


def test_compare_materials():
    response = client.get("/api/v1/samples?count=2")
    assert response.status_code == 200
    samples = response.json()["samples"]

    compared = client.post(
        "/api/v1/compare",
        json={"material_a": samples[0], "material_b": samples[1]},
    )
    assert compared.status_code == 200
    body = compared.json()
    assert "material_a" in body
    assert "material_b" in body
    assert body["higher_predicted_material"] in {"Material A", "Material B"}


def test_sample_csv_download_and_ranked_batch_prediction():
    sample_csv = client.get("/api/v1/sample.csv")
    assert sample_csv.status_code == 200
    assert "text/csv" in sample_csv.headers["content-type"]

    batch = client.post(
        "/api/v1/predict-batch",
        files={"file": ("sample.csv", io.BytesIO(sample_csv.content), "text/csv")},
    )
    assert batch.status_code == 200
    text = batch.text
    assert "candidate_rank" in text
    assert "predicted_critical_temp_k" in text
    assert "ood_risk_score" in text


def test_batch_rejects_wrong_extension():
    response = client.post(
        "/api/v1/predict-batch",
        files={"file": ("sample.txt", b"bad,data\n1,2", "text/plain")},
    )
    assert response.status_code == 400


def test_live_metrics_and_observability_headers():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] >= 1
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-ms" in response.headers


def test_favicon_robots_and_custom_404():
    assert client.get("/favicon.svg").status_code == 200
    assert client.get("/robots.txt").status_code == 200

    missing = client.get("/definitely-not-a-route", headers={"accept": "text/html"})
    assert missing.status_code == 404
    assert "Return home" in missing.text
