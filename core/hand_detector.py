"""MediaPipe Tasks HandLandmarker wrapper (the deprecated ``mp.solutions`` API is NOT used)."""
from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import load_config, path_of
from core.landmarks import HAND_CONNECTIONS, HandObservation

log = logging.getLogger(__name__)


class DetectorUnavailable(RuntimeError):
    """Raised when the hand model cannot be downloaded/loaded. The message is user-presentable."""


def ensure_hand_model(path: Optional[Path] = None, url: Optional[str] = None, timeout: float = 30.0) -> Path:
    """Return the local ``hand_landmarker.task``; download it on first use."""
    cfg = load_config()
    path = Path(path) if path else path_of("hand_model")
    url = url or cfg["hand_detector"]["model_url"]
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        log.info("Downloading hand landmarker model from %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if tmp.stat().st_size < 1_000_000:
            raise OSError("downloaded file is too small to be a valid model")
        tmp.replace(path)
        return path
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise DetectorUnavailable(
            f"Hand model not available ({exc}). Download it manually from\n  {url}\n"
            f"and save it as\n  {path}") from exc


class HandDetector:
    """Detect up to two hands and return :class:`HandObservation` objects.

    ``mode='video'`` uses MediaPipe's tracking (needs monotonically increasing timestamps) and
    is intended for live streams; ``mode='image'`` is stateless (datasets, snapshots).
    """

    def __init__(self, mode: str = "video", model_path: Optional[Path] = None) -> None:
        import mediapipe as mp                       # imported lazily; see core.compat for order
        from mediapipe.tasks.python import BaseOptions, vision

        cfg = load_config()["hand_detector"]
        self._mp = mp
        self.mode = mode
        self.model_path = ensure_hand_model(model_path)
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running, num_hands=int(cfg["num_hands"]),
            min_hand_detection_confidence=float(cfg["min_detection_confidence"]),
            min_hand_presence_confidence=float(cfg["min_presence_confidence"]),
            min_tracking_confidence=float(cfg["min_tracking_confidence"]))
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(opts)
        except Exception as exc:
            raise DetectorUnavailable(f"Could not load hand model {self.model_path}: {exc}") from exc
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> list[HandObservation]:
        """Detect hands in an RGB uint8 image."""
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            ts = int(timestamp_ms if timestamp_ms is not None else self._last_ts + 33)
            ts = max(ts, self._last_ts + 1)                 # MediaPipe requires strictly increasing
            self._last_ts = ts
            res = self._landmarker.detect_for_video(mp_img, ts)
        else:
            res = self._landmarker.detect(mp_img)
        hands: list[HandObservation] = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            hands.append(HandObservation(pts, handed[0].category_name, float(handed[0].score)))
        return hands

    def close(self) -> None:
        """Release native resources."""
        try:
            self._landmarker.close()
        except Exception:
            pass


def draw_hands(frame_bgr: np.ndarray, hands: list[HandObservation], color=(0, 220, 120)) -> np.ndarray:
    """Draw the skeleton overlay in place and return the frame."""
    h, w = frame_bgr.shape[:2]
    for obs in hands:
        pts = (obs.landmarks[:, :2] * np.array([w, h])).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame_bgr, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, tuple(p), 4, (40, 40, 40), 1, cv2.LINE_AA)
    return frame_bgr


def draw_skeleton_arrays(frame_bgr: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw skeletons from slot arrays (used by the no-camera demo)."""
    hands = [HandObservation(raw[s], "Right" if s == 0 else "Left") for s in range(2) if mask[s]]
    return draw_hands(frame_bgr, hands)
