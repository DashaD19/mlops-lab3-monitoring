"""Інтеграційні тести REST API через TestClient."""
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, REFERENCE_PATH, app, loadModel
from ml.train import trainAndSave

# Гарантуємо існування артефактів перед запуском API-тестів — у CI-середовищі
# репозиторій склонований «начисто», ані model.joblib, ані reference_stats.joblib немає.
if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    trainAndSave(MODEL_PATH, REFERENCE_PATH)
# Стартова подія on_event у нових версіях Starlette не виконується автоматично
# при створенні TestClient у модульній області — викликаємо завантаження явно.
loadModel()

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["drift_detector_ready"] is True


def test_predict_class_0():
    """Типовий зразок класу_0 має класифікуватись саме як class_0."""
    payload = {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127.0,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280_od315_of_diluted_wines": 3.92,
        "proline": 1065.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "class_0"
    assert body["class_id"] == 0
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_input():
    """Рядок замість float має давати 422 від Pydantic-валідатора."""
    payload = {"alcohol": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
