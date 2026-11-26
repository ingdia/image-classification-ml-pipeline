"""Dataset statistics behind the dashboard visualisations: python -m src.dataset_stats

Three features of the scanned documents are measured (on the ORIGINAL images,
before resizing):
  1. Class distribution   - images per document type and split (is the data balanced?)
  2. Page geometry        - width, height and aspect ratio (are documents a consistent shape?)
  3. Ink density / brightness - share of dark pixels and mean brightness per type
     (do document types differ visually, e.g. tables and boxes in forms vs. sparse letters?)
Results are written to data/dataset_stats.json, which is committed so the deployed
dashboard can show them without shipping the full dataset.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, DATA_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.preprocessing import class_counts, list_labelled_files

STATS_PATH = DATA_DIR / "dataset_stats.json"
DARK_THRESHOLD = 128  # pixel < 128 on a 0-255 scale counts as "ink"


def image_features(path) -> dict:
    with Image.open(path) as img:
        width, height = img.size
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": height / width,
        "brightness": float(gray.mean()),
        "ink_density": float((gray < DARK_THRESHOLD).mean()),
        "file_kb": path.stat().st_size / 1024,
    }


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "median": float(np.median(a)), "max": float(a.max())}


def compute_stats() -> dict:
    splits = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    distribution = {name: class_counts(d) for name, d in splits.items()}

    per_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    for d in splits.values():
        for path, label in list_labelled_files(d):
            per_class[CLASS_NAMES[label]].append(image_features(path))

    features = {}
    for c, rows in per_class.items():
        if rows:
            features[c] = {k: summarise([r[k] for r in rows]) for k in rows[0]}

    all_rows = [r for rows in per_class.values() for r in rows]
    hist_brightness, edges_b = np.histogram([r["brightness"] for r in all_rows], bins=20, range=(0, 255))
    hist_ink, edges_i = np.histogram([r["ink_density"] for r in all_rows], bins=20, range=(0, 0.5))
    return {
        "class_names": CLASS_NAMES,
        "total_images": len(all_rows),
        "distribution": distribution,
        "per_class_features": features,
        "sizes": [{"class": c, "width": r["width"], "height": r["height"]}
                  for c, rows in per_class.items() for r in rows[::4]],  # every 4th image for scatter
        "histograms": {
            "brightness": {"counts": hist_brightness.tolist(), "edges": edges_b.tolist()},
            "ink_density": {"counts": hist_ink.tolist(), "edges": edges_i.tolist()},
        },
    }


if __name__ == "__main__":
    stats = compute_stats()
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {STATS_PATH} ({stats['total_images']} images)")
