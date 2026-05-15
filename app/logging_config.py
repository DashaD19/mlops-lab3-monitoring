"""Структуроване JSON-логування для ML API.

Кожна подія серіалізується у JSON з полями timestamp / level / logger / message
+ довільні extra. Це машинно-читабельний формат, придатний для ELK / Loki /
CloudWatch без додаткового парсингу.
"""
import logging
import sys

from pythonjsonlogger import jsonlogger


def setupLogging(level: int = logging.INFO) -> None:
    """Налаштовує root logger на JSON-формат у stdout."""
    logger = logging.getLogger()
    logger.setLevel(level)

    # Прибираємо дефолтні хендлери uvicorn, щоб уникнути дубльованого виводу
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Глушимо вбудовані access-логи uvicorn — пишемо власні структуровані події
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
