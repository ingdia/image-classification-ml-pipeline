"""Data acquisition (large version): sample our 9 document types from the FULL RVL-CDIP.

RVL-CDIP has 400,000 scans (320k train / 40k val / 40k test, 16 types). It is available on
Kaggle as `abdellatifsassioui/rvlcdip` (layout: labels/{train,val,test}.txt and images/...).
This script samples N images per type from each OFFICIAL split, converts the TIFF scans to
grayscale PNG and writes them as data/{train,val,test}/<type>/*.png - the same layout the rest
of the pipeline (preprocessing, training, statistics) already uses.

Images that also appear in RVL-CDIP small-200 (the set behind scripts/download_data.py, whose
test split the deployed API evaluates on) are excluded, so that set stays unseen.

Usage (on Kaggle the dataset is mounted at /kaggle/input/rvlcdip):
    python scripts/build_rvl_subset.py --rvl-root /kaggle/input/rvlcdip --out data \
        --train 2000 --val 250 --test 250 --exclude-dir data_small200/raw
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASS_NAMES, SEED  # noqa: E402

# Official RVL-CDIP label ids for our 9 types
RVL_IDS = {0: "letter", 1: "form", 2: "email", 3: "handwritten", 10: "budget", 11: "invoice",
           13: "questionnaire", 14: "resume", 15: "memo"}
assert sorted(RVL_IDS.values()) == sorted(CLASS_NAMES)
SPLITS = {"train": "train", "val": "val", "test": "test"}  # our split -> RVL label file


def find_images_dir(root: Path) -> Path:
    """The Kaggle copy stores label paths relative to <root>/images."""
    return root / "images" if (root / "images").is_dir() else root


def read_labels(root: Path, split: str) -> dict[str, list[str]]:
    per_class: dict[str, list[str]] = {c: [] for c in CLASS_NAMES}
    for line in (root / "labels" / f"{split}.txt").read_text().splitlines():
        rel, label = line.rsplit(" ", 1)
        if int(label) in RVL_IDS:
            per_class[RVL_IDS[int(label)]].append(rel)
    return per_class


def _convert(job: tuple[str, str]) -> bool:
    src, dst = job
    try:
        with Image.open(src) as img:
            img.convert("L").save(dst)
        return True
    except Exception as exc:  # a handful of RVL-CDIP scans are corrupt
        print(f"skip {src}: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvl-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--exclude-dir", type=Path, default=None,
                    help="folder whose image file names must not be used (RVL-CDIP small-200)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    excluded = set()
    if args.exclude_dir and args.exclude_dir.exists():
        excluded = {p.stem for p in args.exclude_dir.rglob("*") if p.is_file()}
    print(f"excluding {len(excluded)} small-200 images")

    images_dir = find_images_dir(args.rvl_root)
    rng = random.Random(SEED)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    jobs = []
    for split, label_file in SPLITS.items():
        for c, rels in read_labels(args.rvl_root, label_file).items():
            rels = sorted(r for r in rels if Path(r).stem not in excluded)
            rng.shuffle(rels)
            out_dir = args.out / split / c
            out_dir.mkdir(parents=True, exist_ok=True)
            for rel in rels[:counts[split]]:
                jobs.append((str(images_dir / rel), str(out_dir / (Path(rel).stem + ".png"))))

    print(f"converting {len(jobs)} scans with {args.workers} workers ...")
    with ProcessPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(_convert, jobs, chunksize=64))
    print(f"done: {ok}/{len(jobs)} images written to {args.out}")


if __name__ == "__main__":
    main()
