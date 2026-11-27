"""Locust load test: simulated users sending documents to the prediction API.

Each simulated user repeatedly uploads a random sample document (data/samples)
to /api/predict through the nginx load balancer, with an occasional /api/health
check, waiting 0.5-1.5 s between requests like a person using the dashboard.

Run headless (see scripts/run_load_tests.sh), or with the web UI:
    docker compose --profile loadtest run --rm -p 8089:8089 locust -f locustfile.py --host http://nginx
"""
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

# A user gives up after this many seconds; under overload this shows up as a failure
# instead of a request that never completes during the test window.
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30))

SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "../data/samples"))
SAMPLES = [(p.name, p.read_bytes()) for p in sorted(SAMPLES_DIR.glob("*.png"))]
if not SAMPLES:
    raise RuntimeError(f"No sample images found in {SAMPLES_DIR}")


class DocumentUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, content = random.choice(SAMPLES)
        with self.client.post("/api/predict", files={"file": (name, content, "image/png")},
                              name="POST /api/predict", catch_response=True,
                              timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code == 0:
                resp.failure(f"no response within {REQUEST_TIMEOUT:.0f}s")
            elif resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "predicted_class" not in resp.json():
                resp.failure("missing predicted_class")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="GET /api/health", timeout=REQUEST_TIMEOUT)
