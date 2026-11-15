"""Convert uploaded/camera frames to the 28x28 Sign Language MNIST format."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.hand_detector import HandDetector


@dataclass(frozen=True)
class PreparedImage:
    """Model input plus the crop shown to the user."""
    tensor: np.ndarray
    preview: np.ndarray
    bbox: tuple[int, int, int, int]


class ImagePreprocessor:
    """Use MediaPipe Tasks to crop a hand, then normalise it for the CNN."""

    def __init__(self, mode: str = "image") -> None:
        self.detector = HandDetector(mode)

    def close(self) -> None:
        self.detector.close()

    def prepare(self, bgr: np.ndarray, timestamp_ms: int | None = None) -> PreparedImage:
        if bgr is None or bgr.size == 0:
            raise ValueError("The uploaded image is empty.")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = self.detector.detect(rgb, timestamp_ms)
        if not hands:
            raise ValueError("No hand found. Use a well-lit image with one open hand visible.")
        points = np.concatenate([hand.landmarks[:, :2] for hand in hands], axis=0)
        h, w = bgr.shape[:2]
        x1, y1 = np.floor(points.min(axis=0) * (w, h)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0) * (w, h)).astype(int)
        pad = max(8, int(0.25 * max(x2 - x1, y2 - y1)))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("The detected hand crop is invalid.")
        crop = bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = gray.astype(np.float32)[None, ..., None] / 255.0
        return PreparedImage(tensor, gray, (x1, y1, x2, y2))

