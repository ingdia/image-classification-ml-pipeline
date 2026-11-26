"""Model creation, training and evaluation.

Architecture (transfer learning):
    input uint8 224x224x3
      -> data augmentation (only active during training): small rotation, shift, zoom
         - imitates crooked / misaligned scans. No flips: mirrored documents don't exist.
      -> Rescaling to [-1, 1] (MobileNetV2's expected input range)
      -> MobileNetV2 backbone pretrained on ImageNet (feature extractor)
      -> GlobalAveragePooling -> Dropout(0.3) -> Dense(9, softmax)

Training happens in two stages:
  1. Feature extraction: backbone frozen, only the new classifier head learns (lr 1e-3).
  2. Fine-tuning: the top backbone layers are unfrozen and trained with a very small
     learning rate (1e-5) so they adapt from natural photos to document layouts
     without destroying the pretrained features.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)

from src.config import CLASS_NAMES, IMG_SIZE, LEARNING_RATE, NUM_CLASSES, SEED

FINE_TUNE_LAYERS = int(os.getenv("FINE_TUNE_LAYERS", 30))
FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", 1e-5))


def build_model(weights: str | None = "imagenet") -> tf.keras.Model:
    tf.keras.utils.set_random_seed(SEED)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02, fill_mode="constant", fill_value=255.0),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=255.0),
        tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=255.0),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights=weights)
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize")(x)
    x = base(x, training=False)  # keep BatchNorm statistics frozen
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="doc_classifier")
    compile_model(model, LEARNING_RATE)
    return model


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = FINE_TUNE_LAYERS) -> None:
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:  # BatchNorm layers stay frozen even when fine-tuning
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, FINE_TUNE_LR)


def _callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                                 min_lr=1e-6, verbose=1)]


def train(model, x_train, y_train, x_val, y_val, epochs: int, fine_tune_epochs: int = 0,
          batch_size: int = 32) -> dict:
    """Two-stage training. Returns the merged per-epoch history."""
    history: dict[str, list] = {}

    def _merge(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)

    _merge(model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=batch_size, callbacks=_callbacks(), verbose=2))
    history["fine_tune_start_epoch"] = len(history["accuracy"])
    if fine_tune_epochs > 0:
        unfreeze_top_layers(model)
        _merge(model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         epochs=fine_tune_epochs, batch_size=batch_size,
                         callbacks=_callbacks(), verbose=2))
    return history


def evaluate(model, x, y) -> dict:
    """Accuracy, macro precision/recall/F1, per-class report and confusion matrix."""
    probs = model.predict(x, batch_size=64, verbose=0)
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0)
    labels = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "n_samples": int(len(y)),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y, y_pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "class_names": CLASS_NAMES,
    }


def save_model(model: tf.keras.Model, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)  # .h5 extension -> HDF5 format (required by the assignment brief)
    return path


def load_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                                     if isinstance(o, np.ndarray) else str(o)))
