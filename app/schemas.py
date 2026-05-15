"""Pydantic-моделі для контракту API.

WineFeatures повторюють порядок і назви feature_names з
sklearn.datasets.load_wine() — це гарантує, що np.array з полів схеми
відповідає очікуванням навченого Pipeline. DriftRequest / DriftResponse
описують контракт нового ендпоінта `/check-drift` із ЛР3.
"""
from pydantic import BaseModel, Field, conlist


class WineFeatures(BaseModel):
    """13 фізико-хімічних характеристик зразка вина."""

    alcohol: float = Field(..., ge=0, description="Вміст алкоголю, % об.")
    malic_acid: float = Field(..., ge=0, description="Яблучна кислота, г/л")
    ash: float = Field(..., ge=0, description="Зола, г/л")
    alcalinity_of_ash: float = Field(..., ge=0, description="Лужність золи")
    magnesium: float = Field(..., ge=0, description="Магній, мг/л")
    total_phenols: float = Field(..., ge=0, description="Загальні феноли")
    flavanoids: float = Field(..., ge=0, description="Флавоноїди")
    nonflavanoid_phenols: float = Field(..., ge=0, description="Не-флавоноїдні феноли")
    proanthocyanins: float = Field(..., ge=0, description="Проантоціани")
    color_intensity: float = Field(..., ge=0, description="Інтенсивність кольору")
    hue: float = Field(..., ge=0, description="Відтінок")
    od280_od315_of_diluted_wines: float = Field(..., ge=0, description="OD280/OD315 розведених вин")
    proline: float = Field(..., ge=0, description="Пролін, мг/л")


class PredictionResponse(BaseModel):
    """Відповідь /predict: ідентифікатор класу, його назва, ймовірність."""

    class_id: int
    class_name: str
    probability: float


class DriftRequest(BaseModel):
    """Батч даних для перевірки drift. Кожен sample = 13 числових ознак (порядок Wine)."""

    samples: conlist(conlist(float, min_length=13, max_length=13), min_length=10) = Field(
        ..., description="Не менше 10 спостережень — інакше KS-тест статистично некоректний."
    )
    alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="Поріг значущості")


class FeatureDriftInfo(BaseModel):
    """Деталі KS-тесту для однієї ознаки."""

    statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    """Підсумок перевірки drift по всьому батчу."""

    drift_detected: bool
    n_drifted_features: int
    drifted_features: list[str]
    per_feature: dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
