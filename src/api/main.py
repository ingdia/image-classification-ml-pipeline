"""FastAPI backend for the document classification pipeline.

Run locally:  uvicorn src.api.main:app --reload
Swagger docs: http://localhost:8000/docs   (or http://localhost:8080/docs through nginx)
"""
from __future__ import annotations

import json
import socket
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src import config, registry, retrain, upload_validation
from src.config import ALLOWED_EXTENSIONS, CLASS_NAMES, MAX_UPLOAD_MB
from src.model import evaluate
from src.prediction import ModelNotReadyError, get_predictor
from src.preprocessing import InvalidImageError, is_allowed_file, load_dataset

START_TIME = time.time()
HOSTNAME = socket.gethostname()  # container id -> shows which replica answered

app = FastAPI(
    title="Financial Document Classifier API",
    description=(
        "Classifies scanned documents a lender receives with a credit application into "
        f"{len(CLASS_NAMES)} types ({', '.join(CLASS_NAMES)}). Supports prediction, bulk upload, "
        "data validation, retraining with a promotion gate, model metrics and live evaluation."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = get_predictor()


@app.on_event("startup")
def _warm_up() -> None:
    """Load the model at startup so the first user request is not slow."""
    try:
        predictor.ensure_loaded()
    except ModelNotReadyError:
        pass


async def _read_files(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    return [(f.filename or "unnamed", await f.read()) for f in files]


# ------------------------------------------------------------------ monitoring
@app.get("/health", tags=["monitoring"])
def health():
    """Liveness + uptime of this container and the version of the model it serves."""
    try:
        predictor.ensure_loaded()
        model_ok = True
    except ModelNotReadyError:
        model_ok = False
    uptime = time.time() - START_TIME
    return {
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_version": predictor.version,
        "uptime_seconds": round(uptime, 1),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(START_TIME)),
        "container": HOSTNAME,
        "retraining": retrain.is_running(),
    }


# ------------------------------------------------------------------ prediction
@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="One scanned document image")):
    """Predict the document type of one image."""
    if not is_allowed_file(file.filename or ""):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {MAX_UPLOAD_MB} MB")
    started = time.perf_counter()
    try:
        result = predictor.predict(content)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    result["filename"] = file.filename
    result["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["container"] = HOSTNAME
    return result


# ------------------------------------------------------------------ data upload
@app.post("/upload", tags=["data"], status_code=status.HTTP_201_CREATED)
async def upload(files: list[UploadFile] = File(..., description="Images and/or ZIP archives"),
                 label: str | None = Form(None, description=f"Document type: one of {CLASS_NAMES}. "
                                                            "Optional for ZIPs with <class>/ folders.")):
    """Bulk upload of labelled images for retraining. Invalid files are rejected with a reason."""
    results = upload_validation.save_files(await _read_files(files), label)
    summary = upload_validation.summarise(results)
    if summary["valid"] == 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"message": "No valid images in upload", **summary})
    summary["pending_uploads"] = upload_validation.uploads_summary()
    return summary


@app.post("/validate", tags=["data"])
async def validate(files: list[UploadFile] = File(...), label: str | None = Form(None)):
    """Dry run: check files without saving them (decodable, allowed type, size, known label)."""
    return upload_validation.summarise(upload_validation.validate_files(await _read_files(files), label))


@app.get("/uploads", tags=["data"])
def uploads():
    """Images currently waiting to be used by the next retraining."""
    return upload_validation.uploads_summary()


@app.delete("/uploads", tags=["data"])
def clear_uploads():
    if retrain.is_running():
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot clear uploads while retraining is running")
    return {"removed": upload_validation.clear_uploads()}


@app.get("/dataset/stats", tags=["data"])
def dataset_stats():
    """Precomputed dataset features for the dashboard visualisations."""
    path = config.DATA_DIR / "dataset_stats.json"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run python -m src.dataset_stats first")
    return json.loads(path.read_text())


@app.get("/classes", tags=["data"])
def classes():
    return {"classes": CLASS_NAMES}


@app.get("/samples", tags=["data"])
def list_samples():
    """Example test documents (2 per type) the dashboard can use for demo predictions."""
    files = sorted(p.name for p in config.SAMPLES_DIR.glob("*.png")) if config.SAMPLES_DIR.exists() else []
    return {"samples": [{"name": f, "label": f.split("_")[0]} for f in files]}


@app.get("/samples/{name}", tags=["data"])
def get_sample(name: str):
    path = (config.SAMPLES_DIR / name).resolve()
    if path.parent != config.SAMPLES_DIR.resolve() or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sample not found")
    return FileResponse(path, media_type="image/png")


# ------------------------------------------------------------------ retraining
def _retrain_job():
    try:
        retrain.run_retraining()
    except Exception as exc:  # status file already records the failure
        print(f"[retrain] failed: {exc}")


@app.post("/retrain", tags=["retraining"], status_code=status.HTTP_202_ACCEPTED)
def trigger_retrain():
    """Start retraining on uploaded data in the background. Poll /retrain/status for progress."""
    if retrain.is_running():
        raise HTTPException(status.HTTP_409_CONFLICT, "A retraining job is already running")
    pending = upload_validation.uploads_summary()
    if pending["total"] == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "No uploaded data. Upload labelled images with POST /upload first.")
    if registry.current_model_info() is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No live model to retrain from")
    retrain.write_status(state="queued", message=f"Retraining queued with {pending['total']} new images")
    threading.Thread(target=_retrain_job, daemon=True).start()
    return {"message": "Retraining started", "new_images": pending, "status_url": "/retrain/status"}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status():
    return {**retrain.read_status(), "running": retrain.is_running()}


# ------------------------------------------------------------------ model performance
@app.get("/metrics", tags=["model"])
def metrics():
    """Test-set metrics of the live model, training curves and the full version history."""
    reg = registry.read_registry()
    out = {"current": reg.get("current"), "history": reg.get("history", [])}
    for name in ("metrics", "history"):
        path = registry.MODELS_DIR / f"{name}.json"
        out["test_metrics" if name == "metrics" else "training_history"] = (
            json.loads(path.read_text()) if path.exists() else None)
    return out


@app.post("/evaluate", tags=["model"])
def evaluate_live_model():
    """Evaluation in production: score the model that is live RIGHT NOW on the held-out test set."""
    try:
        model = predictor.ensure_loaded()
    except ModelNotReadyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    x_test, y_test = load_dataset(config.TEST_DIR)
    if len(x_test) == 0:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Test set not available on this server")
    started = time.perf_counter()
    result = evaluate(model, x_test, y_test)
    result.update(model_version=predictor.version, container=HOSTNAME,
                  evaluated_at=registry.now_iso(),
                  evaluation_seconds=round(time.perf_counter() - started, 1))
    return result
