![CI](https://github.com/DashaD19/mlops-lab3-monitoring/actions/workflows/ci.yml/badge.svg)

# Wine ML API + Monitoring — лабораторна робота № 3 «Моніторинг ML API та детекція проблем»

Курс «MLOps у веб-системах», магістратура КПІ, 10 семестр. Варіант 2.

## Опис системи

Проєкт є розширенням ML API з [`mlops-lab2-ml-api`](https://github.com/DashaD19/mlops-lab2-ml-api) шаром повноцінного спостереження (observability) та автоматичної детекції зсуву даних (data drift). На основі того самого Pipeline зі StandardScaler + SVC (датасет Wine з `scikit-learn`, варіант 2) додано:

- `app/metrics.py` — 7 метрик Prometheus у власному `CollectorRegistry`: лічильники запитів і помилок, гістограми latency та confidence, gauge стану моделі, лічильники перевірок та виявлень drift;
- `GET /metrics` — endpoint у Prometheus exposition format, який автоматично віддає всі зареєстровані метрики;
- `app/drift.py` — клас `DriftDetector` з KS-тестом (`scipy.stats.ks_2samp`) по кожній з 13 ознак Wine;
- `POST /check-drift` — приймає батч живих спостережень (≥ 10 семплів × 13 ознак) і повертає прапор `drift_detected` разом з повним per-feature розкладом;
- `app/logging_config.py` — структуроване JSON-логування (`python-json-logger`) для трьох ключових подій: `startup`, `prediction`, `drift_check`;
- `monitoring/prometheus.yml` + `monitoring/docker-compose.monitoring.yml` — повний локальний стек Prometheus + ML API з scrape-інтервалом 10 секунд;
- `scripts/evidently_report.py` — bonus-завдання: HTML-звіт Evidently для візуального розслідування drift.

CI/CD з ЛР2 успадковано без змін (тести і docker-build пройдуть із новими тестами автоматично); деплой залишається на безкоштовному тарифі Render.

## Реалізовані метрики

| Назва | Тип | Мітки | Призначення |
|---|---|---|---|
| `ml_predictions_total` | Counter | `class_name`, `status` | Кількість прогнозів за класом і статусом — основа для `rate()` PromQL-запитів |
| `ml_prediction_latency_seconds` | Histogram | — | Розподіл latency інференсу (buckets від 5 мс до 5 с); `histogram_quantile(0.95, ...)` дає p95 |
| `ml_prediction_confidence` | Histogram | — | Розподіл `predict_proba` обраного класу (buckets 0.1 → 1.0) — індикатор «впевненості» моделі |
| `ml_errors_total` | Counter | `error_type` | Помилки за типом (`model_not_loaded`, `inference_error`, `drift_detector_not_ready`) |
| `ml_model_loaded` | Gauge | — | 1 = модель завантажена, 0 = ні; миттєвий індикатор стану сервісу |
| `ml_drift_checks_total` | Counter | — | Загальна кількість викликів `/check-drift` |
| `ml_drift_detected_total` | Counter | `feature` | Кількість випадків, коли drift був виявлений для конкретної ознаки |

Latency `/predict` фіксується middleware'ом (`metricsMiddleware` у `app/main.py`), що міряє повний цикл HTTP-запиту до цього ендпоінта. Решта метрик інкрементуються безпосередньо у тілі обробників.

Корисні PromQL-запити для перевірки:

```promql
rate(ml_predictions_total[1m])                                    # RPS прогнозів
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))  # p95 latency
sum by (class_name) (ml_predictions_total)                        # розподіл за класами
histogram_quantile(0.5, rate(ml_prediction_confidence_bucket[5m]))  # медіана confidence
rate(ml_errors_total[5m])                                         # error rate за типами
ml_drift_detected_total                                           # сумарно виявлено drift по ознаках
```

## Drift detection

Детектор реалізує статистичний підхід — двовибірковий **KS-тест Колмогорова-Смирнова** проти reference-вибірки (X_train з тренування, збережена у `reference_stats.joblib`).

Алгоритм для кожної з 13 ознак:

1. `scipy.stats.ks_2samp(reference[:, i], current[:, i])` повертає `(D, p_value)`.
2. Якщо `p_value < alpha` (за замовчуванням `0.05`) — для цієї ознаки drift зафіксовано.
3. Загальний `drift_detected = OR` за всіма ознаками; додатково повертається список `drifted_features` та per-feature деталі.

Обмеження KS-тесту враховано на рівні валідації Pydantic-схеми `DriftRequest`: `samples` має містити мінімум 10 спостережень × рівно 13 числових ознак (нижче — статистично некоректно).

## Приклади запитів

### Прогноз — `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
    "alcalinity_of_ash": 15.6, "magnesium": 127.0, "total_phenols": 2.8,
    "flavanoids": 3.06, "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29, "color_intensity": 5.64, "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0
  }'
# {"class_id":0,"class_name":"class_0","probability":0.9878}
```

### Перевірка drift — `POST /check-drift`

**Сценарій 1 — «здоровий» батч (значення, близькі до reference):**

```bash
curl -X POST http://localhost:8000/check-drift \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065],
      [13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050],
      [13.16,2.36,2.67,18.6,101,2.8,3.24,0.3,2.81,5.68,1.03,3.17,1185],
      [14.37,1.95,2.5,16.8,113,3.85,3.49,0.24,2.18,7.8,0.86,3.45,1480],
      [13.24,2.59,2.87,21,118,2.8,2.69,0.39,1.82,4.32,1.04,2.93,735],
      [14.2,1.76,2.45,15.2,112,3.27,3.39,0.34,1.97,6.75,1.05,2.85,1450],
      [14.39,1.87,2.45,14.6,96,2.5,2.52,0.3,1.98,5.25,1.02,3.58,1290],
      [14.06,2.15,2.61,17.6,121,2.6,2.51,0.31,1.25,5.05,1.06,3.58,1295],
      [14.83,1.64,2.17,14,97,2.8,2.98,0.29,1.98,5.2,1.08,2.85,1045],
      [13.86,1.35,2.27,16,98,2.98,3.15,0.22,1.85,7.22,1.01,3.55,1045]
    ],
    "alpha": 0.05
  }'
```

На таких маленьких батчах (10 семплів проти 142 у reference) p-value для деяких ознак ситуативно може опуститися нижче 0.05 — це властивість KS-тесту, а не помилка детектора. У продакшені рекомендований розмір live-вибірки — 100–1000+ для статистично надійних висновків.

**Сценарій 2 — drifted батч (свідомо аномальні значення):**

```bash
curl -X POST http://localhost:8000/check-drift \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      [20,8,5,40,300,10,12,2,8,20,3,8,3000],
      [21,8.5,5.2,42,310,10.5,12.5,2.1,8.5,21,3.1,8.2,3100],
      [20.5,8.2,5.1,41,305,10.2,12.2,2.05,8.2,20.5,3.05,8.1,3050],
      [22,9,5.5,45,320,11,13,2.2,9,22,3.2,8.5,3200],
      [21.5,8.8,5.4,44,315,10.8,12.8,2.15,8.8,21.5,3.15,8.4,3150],
      [20.2,8.1,5.05,40.5,302,10.1,12.1,2.02,8.1,20.2,3.02,8.05,3020],
      [21.2,8.3,5.15,41.5,307,10.3,12.3,2.07,8.3,20.7,3.07,8.15,3070],
      [22.5,9.2,5.6,46,325,11.2,13.2,2.25,9.2,22.5,3.25,8.6,3250],
      [21.8,8.7,5.35,43,312,10.7,12.7,2.12,8.7,21.2,3.12,8.3,3120],
      [20.8,8.4,5.25,42,309,10.4,12.4,2.09,8.4,20.8,3.09,8.2,3090]
    ],
    "alpha": 0.05
  }'
# {"drift_detected":true,"n_drifted_features":13,"drifted_features":[...all 13...],...}
```

Очікувано: `drift_detected: true`, `n_drifted_features: 13`, для кожної ознаки `p_value` мізерно мале — модель майже напевно бачить «не свої» дані.

## Логування

Усі службові події серіалізуються у JSON у `stdout`. Це формат, придатний для агрегаторів (ELK, Loki, CloudWatch) без додаткового парсингу.

Три ключові події:

- `event=startup` — публікується один раз при завантаженні моделі + reference. Поля: `model_loaded`, `drift_detector_ready`.
- `event=prediction` — на кожен успішний `/predict`. Поля: `class_id`, `class_name`, `probability`.
- `event=drift_check` — на кожен `/check-drift`. Поля: `n_samples`, `alpha`, `drift_detected`, `drifted_features`.

Помилкові гілки (`inference_error`, тощо) логуються через `logger.exception` із повним стеком.

Приклад одного `prediction`-рядка:

```json
{"timestamp":"2026-05-16 10:00:01,234","level":"INFO","logger":"ml-api","message":"prediction","event":"prediction","class_id":0,"class_name":"class_0","probability":0.9878}
```

Окремо у `app/logging_config.py` понижено рівень `uvicorn.access` до `WARNING` — щоб дублюючі неструктуровані access-логи не засмічували stdout.

## Як запустити моніторинг

### Локально без Prometheus

```bash
git clone https://github.com/DashaD19/mlops-lab3-monitoring.git
cd mlops-lab3-monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m ml.train                         # створює model.joblib + reference_stats.joblib
uvicorn app.main:app --reload
```

Тоді відкрити `http://localhost:8000/docs` — у Swagger UI з'являться нові ендпоінти `GET /metrics` і `POST /check-drift`, а консоль почне виводити JSON-логи.

### Локально з Prometheus (docker-compose)

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up --build
```

Після білду:

- ML API — `http://localhost:8000` (`/`, `/health`, `/predict`, `/metrics`, `/check-drift`, `/docs`)
- Prometheus — `http://localhost:9090`
  - `/targets` — статус scrape (`ml-api:8000` має бути **UP**)
  - `/graph` — інтерактивний PromQL-редактор для запитів вище

Згенерувати навантаження для перевірки графіків:

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"alcohol":14.23,"malic_acid":1.71,"ash":2.43,"alcalinity_of_ash":15.6,"magnesium":127,"total_phenols":2.8,"flavanoids":3.06,"nonflavanoid_phenols":0.28,"proanthocyanins":2.29,"color_intensity":5.64,"hue":1.04,"od280_od315_of_diluted_wines":3.92,"proline":1065}' \
    > /dev/null
done
```

Тоді у Prometheus виконати `rate(ml_predictions_total[1m])` — графік покаже зростання частоти прогнозів.

### Запуск тестів

```bash
pytest -q
```

Очікуваний вивід — **11 passed**: 4 тести з ЛР2 (`test_root_endpoint`, `test_health_endpoint`, `test_predict_class_0`, `test_predict_invalid_input`) + 3 тести моделі (`test_train_creates_model_and_reference`, `test_model_predicts_three_classes`, `test_reference_contains_feature_names`) + 2 нові тести метрик + 2 нові тести drift.

### Bonus: Evidently HTML-звіт

```bash
python scripts/evidently_report.py
```

Скрипт читає `reference_stats.joblib`, штучно зсуває `alcohol` і `color_intensity` у current-вибірці та зберігає інтерактивний звіт у `drift_report.html`. Зручний інструмент для post-mortem розслідувань.

### Деплой на Render

Сервіс розгорнуто на безкоштовному тарифі Render через Docker:

- **Публічний URL:** https://mlops-lab3-monitoring.onrender.com
- **Liveness:** https://mlops-lab3-monitoring.onrender.com/health
- **Метрики:** https://mlops-lab3-monitoring.onrender.com/metrics
- **Swagger UI:** https://mlops-lab3-monitoring.onrender.com/docs

> Free tier «засинає» після ≈ 15 хвилин відсутності трафіку, перший запит після паузи може зайняти 30–60 секунд (cold start).

## Висновки

У ході роботи реалізовано шар спостереження для ML-сервісу, що дозволяє контролювати як технічні характеристики (latency, throughput, error rate), так і характеристики, специфічні саме для машинного навчання (розподіл прогнозів за класами, confidence моделі, статистична відмінність вхідних даних від reference). Завдяки pull-моделі Prometheus та exposition-format на стороні ML API цілий стек моніторингу піднімається однією командою `docker-compose up`, а скрапер автоматично виявляє недоступність сервісу.

Виявлення data drift через KS-тест по кожній з 13 ознак Wine забезпечує ранній сигнал про розбіжність розподілу вхідних даних із тренувальною вибіркою — це той клас деградації моделі, який неможливо побачити у звичайних логах 5xx-помилок. Структуровані JSON-логи додають третій рівень спостережуваності і легко інтегруються у централізовані системи агрегації.

Architecture з ЛР2 (FastAPI + Docker + GitHub Actions + Render) розширена без зломів зворотної сумісності: усі 4 тести з ЛР2 проходять без змін, тренування й деплой залишаються одним кліком, а нові ендпоінти `/metrics` і `/check-drift` повністю самодокументуються через Swagger UI.
