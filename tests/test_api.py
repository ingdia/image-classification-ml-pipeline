import time

import pytest
from fastapi.testclient import TestClient

from src import retrain
from src.api.main import app
from src.config import CLASS_NAMES
from tests.conftest import make_image_bytes


@pytest.fixture
def client(isolated_env):
    return TestClient(app)


def test_health_reports_uptime_and_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == 1 and body["uptime_seconds"] >= 0


def test_predict_valid_image(client):
    r = client.post("/predict", files={"file": ("doc.png", make_image_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CLASS_NAMES and 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image_extension(client):
    r = client.post("/predict", files={"file": ("doc.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_corrupt_image(client):
    r = client.post("/predict", files={"file": ("doc.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_upload_then_validate_and_list(client):
    files = [("files", (f"{i}.png", make_image_bytes(), "image/png")) for i in range(2)]
    files.append(("files", ("bad.png", b"junk", "image/png")))
    r = client.post("/upload", files=files, data={"label": "invoice"})
    assert r.status_code == 201
    assert r.json()["valid"] == 2 and r.json()["invalid"] == 1
    assert client.get("/uploads").json()["per_class"]["invoice"] == 2


def test_upload_with_no_valid_files_is_422(client):
    r = client.post("/upload", files=[("files", ("bad.png", b"junk", "image/png"))],
                    data={"label": "invoice"})
    assert r.status_code == 422


def test_validate_is_dry_run(client):
    r = client.post("/validate", files=[("files", ("a.png", make_image_bytes(), "image/png"))],
                    data={"label": "memo"})
    assert r.json()["valid"] == 1
    assert client.get("/uploads").json()["total"] == 0


def test_retrain_requires_uploads(client):
    assert client.post("/retrain").status_code == 400


def test_retrain_runs_in_background(client, monkeypatch):
    monkeypatch.setattr("src.config.RETRAIN_EPOCHS", 1)
    monkeypatch.setattr("src.config.BATCH_SIZE", 4)
    client.post("/upload", files=[("files", ("a.png", make_image_bytes(), "image/png"))],
                data={"label": "invoice"})
    r = client.post("/retrain")
    assert r.status_code == 202
    for _ in range(120):
        state = client.get("/retrain/status").json()["state"]
        if state in ("promoted", "rejected", "failed"):
            break
        time.sleep(1)
    assert state in ("promoted", "rejected")


def test_retrain_conflict_when_running(client):
    retrain.acquire_lock()
    try:
        assert client.post("/retrain").status_code == 409
    finally:
        retrain.release_lock()


def test_metrics_and_evaluate(client):
    assert client.get("/metrics").json()["current"]["version"] == 1
    body = client.post("/evaluate").json()
    assert body["n_samples"] == 6 and body["model_version"] == 1
    assert len(body["confusion_matrix"]) == len(CLASS_NAMES)
