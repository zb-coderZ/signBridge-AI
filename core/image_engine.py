"""Image, video, and webcam recognition state built on the image model."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core.image_models import ImageModel
from core.image_preprocess import ImagePreprocessor, PreparedImage
from core.sentence_builder import SentenceBuilder
from core.stability import StabilityFilter


@dataclass
class ImageResult:
    prediction: str = ""
    confidence: float = 0.0
    top3: list[tuple[str, float]] | None = None
    prepared: PreparedImage | None = None
    error: str = ""
    latency_ms: float = 0.0
    committed: str = ""


class ImageRecognitionEngine:
    """Thread-safe image recognizer shared by Streamlit uploads and WebRTC."""

    def __init__(self, model: ImageModel, threshold: float = 0.65, stable_frames: int = 5, cooldown_s: float = 1.2):
        self.model = model
        self.preprocessor: ImagePreprocessor | None = None
        self.stability = StabilityFilter(stable_frames, threshold, cooldown_s, "unknown", True)
        self.sentence = SentenceBuilder()
        self.last = ImageResult()
        self.history: list[dict] = []
        self.lock = threading.RLock()
        self.on_commit = []
        self.speak_commits = False

    def process(self, bgr: np.ndarray, timestamp_ms: int | None = None, model_name: str = "cnn") -> ImageResult:
        with self.lock:
            started = time.perf_counter()
            try:
                if self.preprocessor is None:
                    self.preprocessor = ImagePreprocessor("video")
                prepared = self.preprocessor.prepare(bgr, timestamp_ms)
                top3, _ = self.model.predict(prepared.tensor, model_name)
                label, confidence = top3[0]
                committed = self.stability.update(label, confidence, time.monotonic()) or ""
                if committed:
                    self.sentence.add(committed)
                    self.history.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "sign": committed,
                                         "confidence": round(confidence, 4), "source": "image/video/webcam"})
                    for callback in self.on_commit:
                        callback(committed, confidence)
                self.last = ImageResult(label, confidence, top3, prepared, "", (time.perf_counter() - started) * 1000, committed)
            except Exception as exc:
                self.stability.reset()
                self.last = ImageResult(error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)
            return self.last

    def reset(self) -> None:
        with self.lock:
            self.stability.reset()
            self.sentence.clear()
            self.history.clear()

    def close(self) -> None:
        if self.preprocessor is not None:
            self.preprocessor.close()

