import pytest

from src import registry
from src.config import CLASS_NAMES
from src.model import build_model, save_model
from src.prediction import ModelNotReadyError, Predictor


def test_prediction_output_structure(isolated_registry, image_bytes):
    result = Predictor().predict(image_bytes)
    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == set(CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["model_version"] == 1


def test_confidence_is_max_probability(isolated_registry, image_bytes):
    r = Predictor().predict(image_bytes)
    assert r["confidence"] == pytest.approx(max(r["probabilities"].values()))
    assert r["probabilities"][r["predicted_class"]] == pytest.approx(r["confidence"])


def test_no_model_raises(tmp_path, monkeypatch, image_bytes):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "missing.json")
    with pytest.raises(ModelNotReadyError):
        Predictor().predict(image_bytes)


def test_hot_reload_after_promotion(isolated_registry, image_bytes):
    predictor = Predictor()
    assert predictor.predict(image_bytes)["model_version"] == 1
    save_model(build_model(weights=None), isolated_registry / "doc_classifier_v2.h5")
    registry.record({"version": 2, "file": "doc_classifier_v2.h5", "status": "promoted",
                     "metrics": {"accuracy": 0.6}}, promote=True)
    assert predictor.predict(image_bytes)["model_version"] == 2
