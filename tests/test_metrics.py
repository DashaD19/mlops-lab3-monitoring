"""Тести Prometheus-метрик: ендпоінт `/metrics` віддає очікувані сімейства,
   виклик `/predict` інкрементує лічильник `ml_predictions_total`."""
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, REFERENCE_PATH, app, loadModel
from ml.train import trainAndSave

# CI клонує репо «начисто» — треба згенерувати model.joblib + reference_stats.joblib
if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    trainAndSave(MODEL_PATH, REFERENCE_PATH)
# У сучасних версіях Starlette startup-хук не тригериться TestClient — викликаємо вручну
loadModel()

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
    "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
    "color_intensity": 5.64, "hue": 1.04, "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065.0,
}


def test_metrics_endpoint_available():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "ml_predictions_total" in body
    assert "ml_prediction_latency_seconds" in body
    assert "ml_prediction_confidence" in body
    assert "ml_model_loaded" in body


def test_predict_increments_counter():
    before = client.get("/metrics").text
    client.post("/predict", json=SAMPLE_PAYLOAD)
    client.post("/predict", json=SAMPLE_PAYLOAD)
    after = client.get("/metrics").text
    # Після двох викликів /predict має з'явитися ряд з міткою success
    assert 'class_name="class_0",status="success"' in after
    assert before != after
