"""Тести KS-детектора drift: відсутність drift при однакових розподілах,
   виявлення drift при зміщенні середнього."""
import numpy as np

from app.drift import DriftDetector

FEATURE_NAMES = ["f1", "f2", "f3", "f4"]


def test_no_drift_on_same_distribution():
    """Дві вибірки з одного N(5,1) — KS-тест НЕ має знайти drift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution():
    """Зміщення середнього з 5 у 8 — drift має бути виявлений на всіх 4 ознаках."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=8.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4
    for feat in FEATURE_NAMES:
        assert result["per_feature"][feat]["p_value"] < 0.05
