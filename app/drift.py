"""Детекція covariate drift на основі двовибіркового KS-тесту.

Для кожної числової ознаки запускається `scipy.stats.ks_2samp` між
reference-вибіркою (X_train, збережена після тренування) та поточною
live-вибіркою з ендпоінта `/check-drift`. Якщо p_value < alpha — для цієї
ознаки drift виявлено; загальний прапор drift_detected = OR за всіма ознаками.
"""
from typing import Any

import numpy as np
from scipy import stats


class DriftDetector:
    """Простий статистичний детектор для числових ознак."""

    def __init__(self, reference: np.ndarray, featureNames: list[str]) -> None:
        if reference.ndim != 2:
            raise ValueError("reference must be 2D (n_samples, n_features)")
        if reference.shape[1] != len(featureNames):
            raise ValueError("featureNames length must match reference columns")
        self.reference = reference
        self.featureNames = featureNames

    def detect(self, current: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
        """KS-тест для кожної ознаки. Повертає словник з повним розкладом."""
        if current.ndim != 2 or current.shape[1] != self.reference.shape[1]:
            raise ValueError(
                f"current must be 2D with {self.reference.shape[1]} columns"
            )

        perFeature: dict[str, dict[str, Any]] = {}
        drifted: list[str] = []

        for i, name in enumerate(self.featureNames):
            refCol = self.reference[:, i]
            curCol = current[:, i]
            ksStat, pValue = stats.ks_2samp(refCol, curCol)
            isDrift = bool(pValue < alpha)
            perFeature[name] = {
                "statistic": float(ksStat),
                "p_value": float(pValue),
                "drift_detected": isDrift,
            }
            if isDrift:
                drifted.append(name)

        return {
            "drift_detected": len(drifted) > 0,
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "per_feature": perFeature,
            "n_samples": int(current.shape[0]),
            "alpha": alpha,
        }
