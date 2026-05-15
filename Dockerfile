# Мінімальний образ Python — менший розмір, швидший pull
FROM python:3.11-slim

# Системні налаштування: не писати .pyc, не буферизувати stdout, не кешувати pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Спочатку копіюємо requirements.txt — окремий шар із залежностями
# залишається у кеші, поки сам файл не змінився.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Тільки після pip install копіюємо застосунковий код:
# зміна одного рядка у app/ не інвалідовуватиме шар із залежностями.
COPY app ./app
COPY ml ./ml

# Тренуємо модель під час збірки, щоб артефакт model.joblib потрапив
# у фінальний образ — на старті контейнера не потрібно нічого довантажувати.
RUN python -m ml.train

# Render передає порт через змінну $PORT; локально — 8000.
ENV PORT=8000
EXPOSE 8000

# sh -c обов'язковий для підстановки $PORT — у exec-формі Docker не розгортає змінні.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
