"""Генерує HTML-звіт Evidently для пари (reference, current).

Bonus до основного online-детектора drift: візуальний інструмент для
розслідування, які саме ознаки «поїхали». Запускається локально:
    python scripts/evidently_report.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "reference_stats.joblib"
OUTPUT_PATH = ROOT / "drift_report.html"


def main() -> None:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{REFERENCE_PATH} не знайдено — спершу запустіть `python -m ml.train`."
        )

    refData = joblib.load(REFERENCE_PATH)
    refDf = pd.DataFrame(refData["X"], columns=refData["feature_names"])

    # Імітуємо drifted live-вибірку: зсуваємо `alcohol` і `color_intensity` для демонстрації
    rng = np.random.default_rng(0)
    current = refDf.sample(n=200, random_state=0, replace=True).reset_index(drop=True)
    current["alcohol"] = current["alcohol"] + rng.uniform(1.0, 2.0, size=len(current))
    current["color_intensity"] = current["color_intensity"] + 2.5

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=refDf, current_data=current)
    report.save_html(str(OUTPUT_PATH))
    print(f"Report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
