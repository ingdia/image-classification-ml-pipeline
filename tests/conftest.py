"""Shared fixtures: tiny synthetic images and an isolated model registry."""
import io

import numpy as np
import pytest
from PIL import Image

from src import registry
from src.config import CLASS_NAMES


def make_image_bytes(fmt="PNG", size=(300, 400), value=None) -> bytes:
    rng = np.random.default_rng(0)
    arr = (np.full((size[1], size[0]), value, dtype=np.uint8) if value is not None
           else rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def image_bytes():
    return make_image_bytes()


@pytest.fixture
def labelled_dir(tmp_path):
    """data dir laid out as <class>/<file>.png with 2 images for 3 classes."""
    for c in CLASS_NAMES[:3]:
        (tmp_path / c).mkdir()
        for i in range(2):
            (tmp_path / c / f"{i}.png").write_bytes(make_image_bytes(value=40 * i + 10))
    return tmp_path


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a temp models dir with an untrained (random) model v1."""
    from src.model import build_model, save_model

    models_dir = tmp_path / "models"
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    monkeypatch.setattr(registry, "REGISTRY_PATH", models_dir / "registry.json")
    save_model(build_model(weights=None), models_dir / "doc_classifier_v1.h5")
    registry.record({"version": 1, "file": "doc_classifier_v1.h5", "status": "promoted",
                     "metrics": {"accuracy": 0.5}}, promote=True)
    return models_dir
