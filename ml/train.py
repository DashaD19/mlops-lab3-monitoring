"""Тренування SVC-класифікатора на Wine з reference-вибіркою для drift detection.

Методичка lab3 пропонує Iris+LogisticRegression як приклад; для варіанта 2
зберігаємо Wine + SVC у Pipeline зі StandardScaler — той самий вибір, що й у ЛР1/ЛР2.

Додатково до моделі зберігається reference_stats.joblib з тренувальною
вибіркою X_train та назвами ознак: саме сирі значення (а не агреговані),
бо KS-тест працює з повним розподілом.
"""
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
REFERENCE_PATH = ROOT / "reference_stats.joblib"


def buildPipeline() -> Pipeline:
    """Pipeline: StandardScaler → SVC(probability=True, random_state=42)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42)),
    ])


def trainAndSave(
    modelPath: Path = MODEL_PATH,
    referencePath: Path = REFERENCE_PATH,
) -> float:
    """Навчає модель на Wine + зберігає reference. Повертає accuracy на test."""
    wine = load_wine()
    X, y = wine.data, wine.target
    featureNames = list(wine.feature_names)

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = buildPipeline()
    model.fit(xTrain, yTrain)
    accuracy = accuracy_score(yTest, model.predict(xTest))

    joblib.dump(model, modelPath)
    joblib.dump(
        {"X": xTrain, "feature_names": featureNames},
        referencePath,
    )
    return accuracy


if __name__ == "__main__":
    acc = trainAndSave()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved model to:     {MODEL_PATH}")
    print(f"Saved reference to: {REFERENCE_PATH}")
