#!/bin/sh
# Container entrypoint for the API.
# On a cloud host with no dataset volume (DOWNLOAD_DATA_ON_START=1), the held-out test set is
# downloaded in the background so "Evaluate live model" and retraining work in production.
# Prediction does not need the dataset, so the API starts serving immediately.
if [ "${DOWNLOAD_DATA_ON_START:-0}" = "1" ] && [ ! -d data/test/budget ]; then
  echo "Downloading dataset in the background ..."
  python scripts/download_data.py > /tmp/download_data.log 2>&1 &
fi
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${UVICORN_WORKERS:-1}"
