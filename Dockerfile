# syntax=docker/dockerfile:1
# Backend image for the document classification pipeline.
#   target "api": production image (FastAPI + TensorFlow), used by docker-compose and the cloud deployment
#   target "dev": api image + Jupyter/pytest, used to train, run the notebook and run tests
FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]

FROM api AS dev
# The dev container runs training/tests, not the API server
HEALTHCHECK NONE
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests/ tests/
COPY notebook/ notebook/
