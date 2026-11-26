import io
import zipfile

from src import upload_validation as uv
from tests.conftest import make_image_bytes


def test_valid_image_accepted():
    assert uv.validate_image("a.png", make_image_bytes(), "invoice").valid


def test_unknown_label_rejected():
    r = uv.validate_image("a.png", make_image_bytes(), "passport")
    assert not r.valid and "unknown label" in r.reason


def test_wrong_extension_rejected():
    r = uv.validate_image("a.pdf", make_image_bytes(), "invoice")
    assert not r.valid and "unsupported" in r.reason


def test_corrupt_image_rejected():
    r = uv.validate_image("a.png", b"not an image", "invoice")
    assert not r.valid and "could not decode" in r.reason


def test_tiny_image_rejected():
    r = uv.validate_image("a.png", make_image_bytes(size=(20, 20)), "invoice")
    assert not r.valid and "too small" in r.reason


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_zip_labels_from_folders():
    z = _zip({"invoice/a.png": make_image_bytes(), "memo/b.png": make_image_bytes(),
              "__MACOSX/._a.png": b"junk"})
    results = uv.validate_files([("batch.zip", z)], label=None)
    assert sorted(r.label for r in results) == ["invoice", "memo"]
    assert all(r.valid for r in results)


def test_flat_zip_uses_request_label():
    z = _zip({"a.png": make_image_bytes(), "b.png": make_image_bytes()})
    results = uv.validate_files([("batch.zip", z)], label="budget")
    assert [r.label for r in results] == ["budget", "budget"]


def test_save_files_writes_only_valid(isolated_env):
    from src import config
    results = uv.save_files([("ok.png", make_image_bytes()), ("bad.png", b"x")], "invoice")
    assert [r.valid for r in results] == [True, False]
    assert len(list((config.UPLOAD_DIR / "invoice").iterdir())) == 1
    assert uv.uploads_summary()["total"] == 1
    assert uv.clear_uploads() == 1
