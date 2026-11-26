"""Model registry: a small JSON file that records which model version is live.

models/registry.json
{
  "current": {"version": 1, "file": "doc_classifier_v1.h5", "metrics": {...}, "created_at": "..."},
  "history": [ ...every version ever trained, promoted or rejected... ]
}

All API containers read this file from a shared volume. When retraining promotes
a new model it rewrites the file atomically, and every container notices the new
version and reloads - so all replicas always serve the same model.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from src.config import MODELS_DIR, REGISTRY_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"current": None, "history": []}
    return json.loads(REGISTRY_PATH.read_text())


def write_registry(registry: dict) -> None:
    """Atomic write: write to a temp file then rename, so readers never see half a file."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=REGISTRY_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def current_model_info() -> dict | None:
    return read_registry().get("current")


def current_model_path():
    info = current_model_info()
    return MODELS_DIR / info["file"] if info else None


def next_version() -> int:
    history = read_registry().get("history", [])
    return max((h["version"] for h in history), default=0) + 1


def record(entry: dict, promote: bool) -> dict:
    registry = read_registry()
    registry.setdefault("history", []).append(entry)
    if promote:
        registry["current"] = entry
    write_registry(registry)
    return registry
