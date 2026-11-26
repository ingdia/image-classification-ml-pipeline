import pytest

from src import config, registry, retrain
from src.upload_validation import save_files
from tests.conftest import make_image_bytes


def _upload_some():
    save_files([(f"{i}.png", make_image_bytes(value=20 * i)) for i in range(3)], "invoice")


def _set_live_accuracy(acc):
    reg = registry.read_registry()
    reg["current"]["metrics"]["accuracy"] = acc
    registry.write_registry(reg)


@pytest.mark.parametrize("cand,cur,expected", [(0.80, 0.80, True), (0.795, 0.80, True),
                                               (0.70, 0.80, False), (0.5, None, True)])
def test_promotion_gate(cand, cur, expected):
    assert retrain.should_promote(cand, cur, tolerance=0.01) is expected


def test_retrain_without_uploads_fails_cleanly(isolated_env):
    with pytest.raises(retrain.NoNewDataError):
        retrain.run_retraining(epochs=1, batch_size=4)
    assert retrain.read_status()["state"] == "failed"
    assert not retrain.is_running()  # lock released


def test_retrain_promotes_better_model(isolated_env):
    _set_live_accuracy(0.0)
    _upload_some()
    entry = retrain.run_retraining(epochs=1, batch_size=4)
    assert entry["status"] == "promoted" and entry["version"] == 2
    assert registry.current_model_info()["version"] == 2
    assert (registry.MODELS_DIR / "doc_classifier_v2.h5").exists()
    assert retrain.read_status()["state"] == "promoted"
    assert entry["uploads_merged"] == 3  # uploads become part of the training set
    assert not any((config.UPLOAD_DIR / "invoice").iterdir())


def test_retrain_rejects_worse_model_and_keeps_live(isolated_env):
    _set_live_accuracy(1.5)  # impossible to reach -> candidate must be rejected
    _upload_some()
    entry = retrain.run_retraining(epochs=1, batch_size=4)
    assert entry["status"] == "rejected"
    assert registry.current_model_info()["version"] == 1  # live model unchanged
    assert (registry.MODELS_DIR / "candidates" / "doc_classifier_v2.h5").exists()
    assert retrain.read_status()["state"] == "rejected"


def test_only_one_retraining_at_a_time(isolated_env):
    retrain.acquire_lock()
    try:
        with pytest.raises(retrain.RetrainBusyError):
            retrain.acquire_lock()
    finally:
        retrain.release_lock()
