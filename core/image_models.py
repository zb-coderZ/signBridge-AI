"""CNN and baseline model construction, persistence, and prediction."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from core.image_dataset import LABEL_TO_LETTER


def build_cnn(n_classes: int):
    import keras
    from keras import layers

    model = keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(96, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_labels(path: Path, labels: list[int]) -> None:
    path.write_text(json.dumps({"labels": labels, "classes": [LABEL_TO_LETTER[x] for x in labels]}, indent=2), encoding="utf-8")


class ImageModel:
    """Thread-safe wrapper around the trained CNN and optional RF baseline."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.cnn = None
        self.rf = None
        self.labels: list[int] = []
        labels_path = self.model_dir / "image_labels.json"
        if labels_path.exists():
            self.labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
        if (self.model_dir / "image_cnn.keras").exists():
            import keras
            self.cnn = keras.models.load_model(self.model_dir / "image_cnn.keras", compile=False)
        if (self.model_dir / "image_rf.joblib").exists():
            self.rf = joblib.load(self.model_dir / "image_rf.joblib")

    @property
    def available(self) -> bool:
        return self.cnn is not None or self.rf is not None

    def predict(self, tensor: np.ndarray, model_name: str = "cnn") -> tuple[list[tuple[str, float]], float]:
        model = self.cnn if model_name == "cnn" else self.rf
        if model is None:
            raise RuntimeError(f"No {model_name} image model is available. Train it first.")
        started = time.perf_counter()
        if model_name == "cnn":
            probs = np.asarray(model.predict(tensor, verbose=0)[0], dtype=np.float32)
        else:
            probs = np.asarray(model.predict_proba(tensor.reshape(1, -1))[0], dtype=np.float32)
        elapsed_ms = (time.perf_counter() - started) * 1000
        order = np.argsort(-probs)[:3]
        return [(LABEL_TO_LETTER[self.labels[int(i)]], float(probs[i])) for i in order], elapsed_ms

