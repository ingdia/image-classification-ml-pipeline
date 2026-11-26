"""Validation and storage of user-uploaded training data.

Every uploaded file must pass these checks before it can be used for retraining:
  1. allowed extension (png, jpg, jpeg, tif, tiff, bmp)
  2. size <= MAX_UPLOAD_MB
  3. decodable as an image (corrupt or fake files are rejected)
  4. at least MIN_SIDE pixels on each side (a thumbnail cannot show a document layout)
  5. label is one of the 9 known document types
ZIP archives are accepted: files inside are labelled by their folder name
(<class>/<file>.png) or, if flat, by the label sent with the request.
"""
from __future__ import annotations

import io
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from src import config
from src.config import ALLOWED_EXTENSIONS, CLASS_NAMES, MAX_UPLOAD_MB

MIN_SIDE = 64


@dataclass
class FileResult:
    filename: str
    label: str | None
    valid: bool
    reason: str = "ok"
    saved_as: str | None = None


def validate_image(filename: str, content: bytes, label: str | None) -> FileResult:
    if label not in CLASS_NAMES:
        return FileResult(filename, label, False, f"unknown label; expected one of {CLASS_NAMES}")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return FileResult(filename, label, False, f"unsupported file type; allowed {sorted(ALLOWED_EXTENSIONS)}")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        return FileResult(filename, label, False, f"file larger than {MAX_UPLOAD_MB} MB")
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.load()
            w, h = img.size
    except (UnidentifiedImageError, OSError):
        return FileResult(filename, label, False, "not a valid image (could not decode)")
    if min(w, h) < MIN_SIDE:
        return FileResult(filename, label, False, f"image too small ({w}x{h}); minimum {MIN_SIDE}px per side")
    return FileResult(filename, label, True)


def expand_files(files: list[tuple[str, bytes]], label: str | None) -> list[tuple[str, bytes, str | None]]:
    """Turn uploads (including ZIP archives) into (filename, content, label) items."""
    items = []
    for name, content in files:
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for info in zf.infolist():
                        path = PurePosixPath(info.filename)
                        if info.is_dir() or path.name.startswith(".") or "__MACOSX" in path.parts:
                            continue
                        folder_label = path.parent.name if path.parent.name in CLASS_NAMES else None
                        items.append((path.name, zf.read(info), folder_label or label))
            except zipfile.BadZipFile:
                items.append((name, b"", label))  # will fail validation with a clear reason
        else:
            items.append((name, content, label))
    return items


def validate_files(files: list[tuple[str, bytes]], label: str | None) -> list[FileResult]:
    """Dry run: validate without saving."""
    return [validate_image(n, c, lbl) for n, c, lbl in expand_files(files, label)]


def save_files(files: list[tuple[str, bytes]], label: str | None) -> list[FileResult]:
    """Validate and store valid files in data/uploads/<label>/ with a unique name."""
    results = []
    for name, content, lbl in expand_files(files, label):
        result = validate_image(name, content, lbl)
        if result.valid:
            target_dir = config.UPLOAD_DIR / lbl
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{uuid.uuid4().hex[:12]}{Path(name).suffix.lower()}"
            target.write_bytes(content)
            result.saved_as = str(target.relative_to(config.DATA_DIR))
        results.append(result)
    return results


def summarise(results: list[FileResult]) -> dict:
    per_class = {c: 0 for c in CLASS_NAMES}
    for r in results:
        if r.valid:
            per_class[r.label] += 1
    return {
        "total": len(results),
        "valid": sum(r.valid for r in results),
        "invalid": sum(not r.valid for r in results),
        "valid_per_class": per_class,
        "files": [asdict(r) for r in results],
    }


def uploads_summary() -> dict:
    """What is currently waiting in data/uploads/ for the next retraining."""
    from src.preprocessing import class_counts
    counts = class_counts(config.UPLOAD_DIR)
    return {"total": sum(counts.values()), "per_class": counts}


def clear_uploads() -> int:
    removed = 0
    for c in CLASS_NAMES:
        d = config.UPLOAD_DIR / c
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
                    removed += 1
    return removed
