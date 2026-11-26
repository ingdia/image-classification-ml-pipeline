"""Retraining pipeline with a promotion gate.

Steps (run in a background thread by POST /retrain, or: python -m src.retrain):
  1. Acquire a lock file on the shared volume -> only one retraining at a time,
     even when several API containers are running.
  2. Load a COPY of the live model (the serving model is never modified in place).
  3. Fine-tune it on original training data + uploaded data (data/uploads/<class>/).
     Old data is kept to avoid "catastrophic forgetting" of the other classes.
  4. Evaluate the candidate on the fixed held-out test set.
  5. Promotion gate: promote only if candidate accuracy >= live accuracy - tolerance.
     Promoted -> registry points to the new version; all containers hot-reload it,
                 and the uploads are merged into data/train for future runs.
     Rejected -> recorded in the registry history; the live model keeps serving.
Progress is written to runtime/retrain_status.json for the dashboard.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config, registry
from src.model import evaluate, load_model, save_json, save_model, unfreeze_top_layers
from src.preprocessing import load_dataset

STALE_LOCK_SECONDS = 2 * 60 * 60


class RetrainBusyError(RuntimeError):
    pass


class NoNewDataError(ValueError):
    pass


# ---------- status + lock (shared between containers via the runtime volume) ----------

def read_status() -> dict:
    if config.RETRAIN_STATUS_PATH.exists():
        try:
            return json.loads(config.RETRAIN_STATUS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"state": "idle", "message": "No retraining has been run yet."}


def write_status(**fields) -> dict:
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    status = {**read_status(), **fields, "updated_at": registry.now_iso()}
    tmp = config.RETRAIN_STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, config.RETRAIN_STATUS_PATH)
    return status


def acquire_lock() -> None:
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lock = config.RETRAIN_LOCK_PATH
    if lock.exists() and time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
        lock.unlink(missing_ok=True)  # left behind by a crashed run
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic: fails if it exists
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError as exc:
        raise RetrainBusyError("A retraining job is already running.") from exc


def release_lock() -> None:
    config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)


def is_running() -> bool:
    return config.RETRAIN_LOCK_PATH.exists()


# ---------- core logic ----------

def should_promote(candidate_acc: float, current_acc: float | None,
                   tolerance: float = config.PROMOTION_TOLERANCE) -> bool:
    if current_acc is None:
        return True
    return candidate_acc >= current_acc - tolerance


class _StatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        write_status(progress={"epoch": epoch + 1, "total_epochs": self.total},
                     message=f"Epoch {epoch + 1}/{self.total}: "
                             f"train acc {logs.get('accuracy', 0):.3f}, val acc {logs.get('val_accuracy', 0):.3f}")


def _merge_uploads_into_train() -> int:
    moved = 0
    for class_dir in config.UPLOAD_DIR.iterdir() if config.UPLOAD_DIR.exists() else []:
        if class_dir.is_dir() and class_dir.name in config.CLASS_NAMES:
            target = config.TRAIN_DIR / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for f in class_dir.iterdir():
                if f.is_file():
                    shutil.move(str(f), target / f"upload_{f.name}")
                    moved += 1
    return moved


def run_retraining(epochs: int | None = None, batch_size: int | None = None) -> dict:
    """Run one full retraining cycle. Returns the registry entry of the candidate."""
    epochs = epochs or config.RETRAIN_EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    acquire_lock()
    start = time.time()
    try:
        write_status(state="running", started_at=registry.now_iso(), finished_at=None,
                     message="Loading data...", progress={"epoch": 0, "total_epochs": epochs},
                     result=None, error=None)
        x_new, y_new = load_dataset(config.UPLOAD_DIR)
        if len(x_new) == 0:
            raise NoNewDataError("No uploaded images found. Upload labelled data before retraining.")
        x_old, y_old = load_dataset(config.TRAIN_DIR)
        x_train = np.concatenate([x_old, x_new]) if len(x_old) else x_new
        y_train = np.concatenate([y_old, y_new]) if len(y_old) else y_new
        x_val, y_val = load_dataset(config.VAL_DIR)
        x_test, y_test = load_dataset(config.TEST_DIR)

        current = registry.current_model_info()
        if current is None:
            raise RuntimeError("No live model to retrain from. Run initial training first.")
        write_status(message=f"Fine-tuning v{current['version']} on {len(x_train)} images "
                             f"({len(x_new)} new)...")
        model = load_model(registry.current_model_path())
        unfreeze_top_layers(model)  # compiles with a small learning rate
        model.fit(x_train, y_train, validation_data=(x_val, y_val) if len(x_val) else None,
                  epochs=epochs, batch_size=batch_size, verbose=2,
                  callbacks=[_StatusCallback(epochs)])

        write_status(message="Evaluating candidate on the held-out test set...")
        metrics = evaluate(model, x_test, y_test)
        current_acc = current.get("metrics", {}).get("accuracy")
        promote = should_promote(metrics["accuracy"], current_acc)

        version = registry.next_version()
        filename = f"doc_classifier_v{version}.h5"
        save_model(model, (registry.MODELS_DIR if promote else config.CANDIDATES_DIR) / filename)
        entry = {
            "version": version,
            "file": filename if promote else f"candidates/{filename}",
            "kind": "retrain",
            "status": "promoted" if promote else "rejected",
            "parent_version": current["version"],
            "created_at": registry.now_iso(),
            "train_samples": int(len(x_train)),
            "new_samples": int(len(x_new)),
            "training_seconds": round(time.time() - start, 1),
            "metrics": {k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "compared_to": {"version": current["version"], "accuracy": current_acc,
                            "tolerance": config.PROMOTION_TOLERANCE},
        }
        if promote:
            save_json(metrics, registry.MODELS_DIR / "metrics.json")
            entry["uploads_merged"] = _merge_uploads_into_train()
        registry.record(entry, promote=promote)

        verdict = (f"PROMOTED: v{version} accuracy {metrics['accuracy']:.3f} "
                   f"vs v{current['version']} {current_acc:.3f}" if promote else
                   f"REJECTED: v{version} accuracy {metrics['accuracy']:.3f} is below "
                   f"v{current['version']} {current_acc:.3f} - tolerance; live model unchanged")
        write_status(state="promoted" if promote else "rejected", message=verdict,
                     finished_at=registry.now_iso(), result=entry)
        return entry
    except Exception as exc:
        write_status(state="failed", message=f"{type(exc).__name__}: {exc}",
                     finished_at=registry.now_iso(), error=traceback.format_exc(limit=3))
        raise
    finally:
        release_lock()


if __name__ == "__main__":
    print(json.dumps(run_retraining(), indent=2))
