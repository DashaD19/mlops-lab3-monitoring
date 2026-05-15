"""Веб-сервіс для інференсу Wine з шаром моніторингу (Prometheus + drift + structlog)."""
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .drift import DriftDetector
from .logging_config import setupLogging
from .metrics import (
    DRIFT_CHECKS,
    DRIFT_DETECTED,
    ERROR_COUNTER,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    REGISTRY,
)
from .schemas import (
    DriftRequest,
    DriftResponse,
    PredictionResponse,
    WineFeatures,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
REFERENCE_PATH = ROOT / "reference_stats.joblib"
CLASS_NAMES = ["class_0", "class_1", "class_2"]

setupLogging()
logger = logging.getLogger("ml-api")

app = FastAPI(
    title="Wine ML API with Monitoring",
    description="REST API для класифікації Wine з Prometheus-метриками та KS-детектором drift",
    version="2.0.0",
)

# Глобальні стани — завантажуються одноразово у startup-хуку.
model = None
driftDetector: DriftDetector | None = None


@app.on_event("startup")
def loadModel() -> None:
    """Завантажує модель та reference-вибірку, ініціалізує DriftDetector."""
    global model, driftDetector
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)

    if REFERENCE_PATH.exists():
        refData = joblib.load(REFERENCE_PATH)
        driftDetector = DriftDetector(
            reference=refData["X"],
            featureNames=refData["feature_names"],
        )
        logger.info(
            "startup_complete",
            extra={"event": "startup", "model_loaded": True, "drift_detector_ready": True},
        )
    else:
        logger.warning(
            "reference_missing",
            extra={"event": "startup", "model_loaded": True, "drift_detector_ready": False},
        )


@app.middleware("http")
async def metricsMiddleware(request: Request, call_next):
    """Універсальний middleware: міряє latency саме `/predict`-запитів."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(elapsed)
    return response


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Wine ML API", "version": "2.0.0"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_detector_ready": driftDetector is not None,
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition endpoint — текст у форматі для scraping."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.array([[
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315_of_diluted_wines,
        features.proline,
    ]])

    try:
        classId = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0, classId])
    except Exception as exc:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.exception("inference_failed", extra={"event": "inference_error"})
        raise HTTPException(status_code=500, detail="Inference error") from exc

    className = CLASS_NAMES[classId]
    PREDICTION_COUNTER.labels(class_name=className, status="success").inc()
    PREDICTION_CONFIDENCE.observe(probability)

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "class_id": classId,
            "class_name": className,
            "probability": round(probability, 4),
        },
    )

    return PredictionResponse(
        class_id=classId,
        class_name=className,
        probability=round(probability, 4),
    )


@app.post("/check-drift", response_model=DriftResponse)
def checkDrift(payload: DriftRequest) -> DriftResponse:
    if driftDetector is None:
        ERROR_COUNTER.labels(error_type="drift_detector_not_ready").inc()
        raise HTTPException(status_code=503, detail="Drift detector is not ready")

    DRIFT_CHECKS.inc()
    current = np.array(payload.samples)
    result = driftDetector.detect(current, alpha=payload.alpha)

    for feat, info in result["per_feature"].items():
        if info["drift_detected"]:
            DRIFT_DETECTED.labels(feature=feat).inc()

    logger.info(
        "drift_check",
        extra={
            "event": "drift_check",
            "n_samples": len(payload.samples),
            "alpha": payload.alpha,
            "drift_detected": result["drift_detected"],
            "drifted_features": result["drifted_features"],
        },
    )

    return DriftResponse(**result)
