"""Data acquisition: download RVL-CDIP small-200 and build train/val/test folders.

Source: https://huggingface.co/datasets/vaclavpechtor/rvl_cdip-small-200
(a 200-images-per-class subset of RVL-CDIP, Harley et al., ICDAR 2015).

We keep only the 9 document types relevant to a credit application (see
src/config.py). The source has 160 "train" + 40 "validation" images per class:
  - source train      -> our train (80%) + val (20%), stratified, fixed seed
  - source validation -> our test (held out, never used for training)
TIFF scans are converted to grayscale PNG because browsers cannot display TIFF.
A few test images per class are copied to data/samples/ for demos.

Usage:  python scripts/download_data.py
"""
import random
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CLASS_NAMES, DATA_DIR, SAMPLES_DIR, SEED, TEST_DIR, TRAIN_DIR, VAL_DIR,
)

REPO_ID = "vaclavpechtor/rvl_cdip-small-200"
RAW_DIR = DATA_DIR / "raw"
VAL_FRACTION = 0.2
SAMPLES_PER_CLASS = 2


def download() -> Path:
    patterns = [f"{split}/{c}/*" for split in ("train", "validation") for c in CLASS_NAMES]
    print(f"Downloading {len(CLASS_NAMES)} classes from {REPO_ID} ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=RAW_DIR,
                      allow_patterns=patterns, max_workers=16)
    return RAW_DIR


def convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("L").save(dst.with_suffix(".png"), optimize=True)


def build_splits(raw: Path) -> None:
    rng = random.Random(SEED)
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR, SAMPLES_DIR):
        for c in CLASS_NAMES:
            shutil.rmtree(d / c, ignore_errors=True)

    for c in CLASS_NAMES:
        train_files = sorted((raw / "train" / c).glob("*"))
        test_files = sorted((raw / "validation" / c).glob("*"))
        rng.shuffle(train_files)
        n_val = int(len(train_files) * VAL_FRACTION)
        splits = {VAL_DIR: train_files[:n_val], TRAIN_DIR: train_files[n_val:], TEST_DIR: test_files}
        for out_dir, files in splits.items():
            for f in files:
                convert(f, out_dir / c / f.name)
        for f in test_files[:SAMPLES_PER_CLASS]:
            convert(f, SAMPLES_DIR / f"{c}_{f.stem}.png")
        print(f"{c:14s} train={len(train_files) - n_val:4d} val={n_val:3d} test={len(test_files):3d}")


if __name__ == "__main__":
    build_splits(download())
    print("Done. Images written to", DATA_DIR)
