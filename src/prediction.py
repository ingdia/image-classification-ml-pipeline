"""Prediction: load the live model and classify a single document image.

The Predictor checks the registry before each prediction (a cheap file stat) and
hot-reloads the model when a retrained version has been promoted, so containers
pick up new models without restarting.

Inference runs through a compiled tf.function instead of calling the Keras model eagerly:
on CPU this is about 6x faster per image (~45 ms vs ~300 ms), which is what lets each
API container serve many requests per second.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import tensorflow as tf

from src import registry
from src.config import CLASS_NAMES
from src.model import load_model
from src.preprocessing import load_image


class ModelNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = None
        self._infer = None
        self.version = None
        self.loaded_at = None
        self._registry_mtime = None

    def _refresh(self) -> None:
        path = registry.REGISTRY_PATH
        if not path.exists():
            return
        mtime = path.stat().st_mtime
        if mtime == self._registry_mtime and self.model is not None:
            return
        with self._lock:
            if mtime == self._registry_mtime and self.model is not None:
                return
            info = registry.current_model_info()
            if info and info["version"] != self.version:
                model = load_model(registry.current_model_path())
                self._infer = _compile(model)
                self.model = model
                self.version = info["version"]
                self.loaded_at = time.time()
            self._registry_mtime = mtime

    def ensure_loaded(self):
        self._refresh()
        if self.model is None:
            raise ModelNotReadyError("No trained model available. Run training first.")
        return self.model

    def predict_array(self, image: np.ndarray) -> dict:
        self.ensure_loaded()
        probs = self._infer(tf.constant(np.expand_dims(image, 0), dtype=tf.uint8)).numpy()[0]
        best = int(np.argmax(probs))
        return {
            "predicted_class": CLASS_NAMES[best],
            "confidence": float(probs[best]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "model_version": self.version,
        }

    def predict(self, source) -> dict:
        """Predict from a file path or raw image bytes."""
        return self.predict_array(load_image(source))


def _compile(model):
    """Compiled inference function (traced once per model, reused for every request)."""
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    infer(tf.zeros((1, *model.input_shape[1:3], 3), dtype=tf.uint8))  # warm-up trace
    return infer


_default = Predictor()


def predict(source) -> dict:
    return _default.predict(source)


def get_predictor() -> Predictor:
    return _default
