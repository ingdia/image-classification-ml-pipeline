"""Data processing: image loading, resizing, label encoding and dataset building.

Pipeline for every image (training, prediction and retraining all use this):
  1. Open with Pillow (works for PNG, JPEG, TIFF, BMP).
  2. Convert to grayscale ("L") - scans are black and white; this removes any colour
     noise from phone photos or colour scanners.
  3. Convert grayscale -> RGB by repeating the channel 3 times, because the
     ImageNet-pretrained MobileNetV2 expects 3 input channels.
  4. Resize to 224x224 (MobileNetV2's native resolution).
  5. Keep pixels as uint8 0-255. Normalisation to [-1, 1] happens INSIDE the
     model (see model.py), so a saved model can never be fed wrongly scaled data.

Labels are encoded as integers using the fixed order of config.CLASS_NAMES
(label encoding); the model outputs one probability per class in that order.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import ALLOWED_EXTENSIONS, CLASS_NAMES, IMG_SIZE


class InvalidImageError(ValueError):
    """Raised when a file cannot be decoded as an image."""


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Load one image (path or raw bytes) as a (224, 224, 3) uint8 array."""
    try:
        img = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    img = img.convert("L").convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def encode_label(name: str) -> int:
    """Label encoding: class name -> integer index."""
    if name not in CLASS_NAMES:
        raise ValueError(f"Unknown class '{name}'. Expected one of {CLASS_NAMES}")
    return CLASS_NAMES.index(name)


def decode_label(index: int) -> str:
    return CLASS_NAMES[int(index)]


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def list_labelled_files(directory: Path) -> list[tuple[Path, int]]:
    """Find images stored as <directory>/<class_name>/<image file>."""
    items = []
    for class_name in CLASS_NAMES:
        class_dir = Path(directory) / class_name
        if class_dir.is_dir():
            items += [(p, encode_label(class_name)) for p in sorted(class_dir.iterdir())
                      if p.is_file() and is_allowed_file(p.name)]
    return items


def load_dataset(directories: Path | Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load every labelled image under one or more directories into memory.

    Returns X with shape (n, 224, 224, 3) uint8 and y with shape (n,) int.
    Undecodable files are skipped with a warning instead of crashing training.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    images, labels = [], []
    for directory in directories:
        for path, label in list_labelled_files(Path(directory)):
            try:
                images.append(load_image(path))
                labels.append(label)
            except InvalidImageError as exc:
                print(f"[preprocessing] skipping {path}: {exc}")
    if not images:
        return np.empty((0, *IMG_SIZE, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def class_counts(directory: Path) -> dict[str, int]:
    """Number of images per class in a directory (used by visualisations and validation)."""
    counts = {c: 0 for c in CLASS_NAMES}
    for _, label in list_labelled_files(Path(directory)):
        counts[CLASS_NAMES[label]] += 1
    return counts
