"""Initial training: python -m src.train

Loads data/train and data/val, trains MobileNetV2 (feature extraction + fine-tuning),
evaluates on the held-out data/test set and registers the result as model v1:
  models/doc_classifier_v1.h5   trained model (HDF5, as required by the brief)
  models/registry.json          live model pointer + version history
  models/metrics.json           test metrics of the live model
  models/history.json           per-epoch training/validation accuracy and loss
"""
from __future__ import annotations

import time

from src import registry
from src.config import BATCH_SIZE, EPOCHS, MODELS_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR
from src.model import build_model, evaluate, save_json, save_model, train
from src.preprocessing import load_dataset

FINE_TUNE_EPOCHS = 8


def main() -> None:
    start = time.time()
    x_train, y_train = load_dataset(TRAIN_DIR)
    x_val, y_val = load_dataset(VAL_DIR)
    x_test, y_test = load_dataset(TEST_DIR)
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    model = build_model()
    history = train(model, x_train, y_train, x_val, y_val, epochs=EPOCHS,
                    fine_tune_epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE)
    metrics = evaluate(model, x_test, y_test)
    print(f"test accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")

    version = registry.next_version()
    filename = f"doc_classifier_v{version}.h5"
    save_model(model, MODELS_DIR / filename)
    save_json(metrics, MODELS_DIR / "metrics.json")
    save_json(history, MODELS_DIR / "history.json")
    registry.record({
        "version": version,
        "file": filename,
        "kind": "initial",
        "status": "promoted",
        "created_at": registry.now_iso(),
        "train_samples": int(len(x_train)),
        "training_seconds": round(time.time() - start, 1),
        "metrics": {k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
    }, promote=True)
    print(f"Saved {filename} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
