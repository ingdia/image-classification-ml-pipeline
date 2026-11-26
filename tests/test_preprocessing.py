import numpy as np
import pytest

from src.config import CLASS_NAMES, IMG_SIZE
from src.preprocessing import (
    InvalidImageError, class_counts, decode_label, encode_label, is_allowed_file, load_dataset,
    load_image,
)
from tests.conftest import make_image_bytes


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "BMP"])
def test_load_image_any_format_gives_224x224x3_uint8(fmt):
    arr = load_image(make_image_bytes(fmt=fmt, size=(500, 700)))
    assert arr.shape == (*IMG_SIZE, 3)
    assert arr.dtype == np.uint8


def test_grayscale_is_repeated_on_three_channels(image_bytes):
    arr = load_image(image_bytes)
    assert np.array_equal(arr[..., 0], arr[..., 1])
    assert np.array_equal(arr[..., 1], arr[..., 2])


def test_load_image_from_path(tmp_path, image_bytes):
    p = tmp_path / "doc.png"
    p.write_bytes(image_bytes)
    assert load_image(p).shape == (*IMG_SIZE, 3)


def test_invalid_bytes_raise_invalid_image_error():
    with pytest.raises(InvalidImageError):
        load_image(b"this is not an image")


def test_label_encoding_round_trip():
    for i, name in enumerate(CLASS_NAMES):
        assert encode_label(name) == i
        assert decode_label(i) == name


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        encode_label("passport")


@pytest.mark.parametrize("name,ok", [("a.png", True), ("b.TIF", True), ("c.jpeg", True),
                                     ("d.pdf", False), ("e.txt", False)])
def test_allowed_file_extensions(name, ok):
    assert is_allowed_file(name) is ok


def test_load_dataset_reads_class_folders(labelled_dir):
    x, y = load_dataset(labelled_dir)
    assert x.shape == (6, *IMG_SIZE, 3)
    assert sorted(set(y.tolist())) == [0, 1, 2]


def test_load_dataset_skips_corrupt_files(labelled_dir):
    (labelled_dir / CLASS_NAMES[0] / "broken.png").write_bytes(b"garbage")
    x, _ = load_dataset(labelled_dir)
    assert len(x) == 6


def test_class_counts(labelled_dir):
    counts = class_counts(labelled_dir)
    assert counts[CLASS_NAMES[0]] == 2 and counts[CLASS_NAMES[-1]] == 0
