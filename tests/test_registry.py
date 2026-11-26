from src import registry


def _use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "registry.json")


def test_empty_registry(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    assert registry.current_model_info() is None
    assert registry.next_version() == 1


def test_rejected_model_is_recorded_but_not_promoted(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    registry.record({"version": 1, "file": "v1.h5", "status": "promoted"}, promote=True)
    registry.record({"version": 2, "file": "v2.h5", "status": "rejected"}, promote=False)
    assert registry.current_model_info()["version"] == 1
    assert [h["version"] for h in registry.read_registry()["history"]] == [1, 2]
    assert registry.next_version() == 3


def test_promotion_switches_current(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    registry.record({"version": 1, "file": "v1.h5"}, promote=True)
    registry.record({"version": 2, "file": "v2.h5"}, promote=True)
    assert registry.current_model_path() == tmp_path / "v2.h5"
