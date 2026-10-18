# Корневой Dockerfile для хостингов, которые ищут его в корне (Render и др.)
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts SPAM_THRESHOLD=0.5
WORKDIR /app
COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src/ /app/src/
COPY service/ /app/service/
COPY scripts/ /app/scripts/

# модели скачиваются из GitHub Releases при сборке: в репозитории только код
RUN python /app/scripts/fetch_models.py /app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
