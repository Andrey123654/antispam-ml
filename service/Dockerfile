# Микросервис антиспам-фильтра: FastAPI + uvicorn
# Python 3.14 — та же мажорная версия, что при обучении модели
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts \
    SPAM_THRESHOLD=0.5

WORKDIR /app

# зависимости ставятся отдельным слоем — кэшируется при изменении кода
COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# код и артефакты модели
COPY src/ /app/src/
COPY service/ /app/service/
COPY scripts/ /app/scripts/

# метрики (маленькие json) копируются, модели — скачиваются из Releases
COPY artifacts/*.json /app/artifacts/
RUN python /app/scripts/fetch_models.py /app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
