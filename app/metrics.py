"""Prometheus-метрики ML API.

Усі метрики оголошуються як модульні змінні і реєструються у власному
CollectorRegistry — це чистіше, ніж глобальний регістр: у тестах легко
скидати стан, а у продакшені не конфліктує з метриками самого FastAPI.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Загальна кількість прогнозів моделі",
    labelnames=["class_name", "status"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Час обробки одного прогнозу (секунди)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Розподіл значень predict_proba для обраного класу",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

ERROR_COUNTER = Counter(
    "ml_errors_total",
    "Кількість помилкових запитів",
    labelnames=["error_type"],
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Чи успішно завантажена модель (1 = так, 0 = ні)",
    registry=REGISTRY,
)

DRIFT_CHECKS = Counter(
    "ml_drift_checks_total",
    "Загальна кількість виконаних перевірок drift",
    registry=REGISTRY,
)

DRIFT_DETECTED = Counter(
    "ml_drift_detected_total",
    "Кількість випадків, коли drift був виявлений",
    labelnames=["feature"],
    registry=REGISTRY,
)
