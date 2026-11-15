"""WebRTC processor for image-model webcam recognition."""
from __future__ import annotations

import queue
import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from core.config import load_config
from core.image_engine import ImageRecognitionEngine


class ImageVideoProcessor(VideoProcessorBase):
    """Return frames immediately and process only the newest frame in a worker."""

    def __init__(self, engine: ImageRecognitionEngine | None = None, model_name: str = "random_forest") -> None:
        self.engine = engine
        self.model_name = model_name
        self._frames: queue.Queue[tuple[object, int]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_processed = 0.0
        self._worker = threading.Thread(target=self._process_frames, name="webcam-inference", daemon=True)
        self._worker.start()

    def _process_frames(self) -> None:
        interval = 1.0 / max(1, int(load_config()["webrtc"]["processing_fps"]))
        while not self._stop.is_set():
            try:
                image, timestamp_ms = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            while True:
                try:
                    image, timestamp_ms = self._frames.get_nowait()
                except queue.Empty:
                    break
            wait = interval - (time.monotonic() - self._last_processed)
            if wait > 0:
                self._stop.wait(wait)
                if self._stop.is_set():
                    return
            if self.engine is not None:
                self.engine.process(image, timestamp_ms=timestamp_ms, model_name=self.model_name)
            self._last_processed = time.monotonic()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        if self.engine is not None:
            timestamp_ms = int(time.monotonic() * 1000)
            try:
                self._frames.put_nowait((image.copy(), timestamp_ms))
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                self._frames.put_nowait((image.copy(), timestamp_ms))
            result = self.engine.last
            if result.prepared is not None:
                x1, y1, x2, y2 = result.prepared.bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def __del__(self) -> None:
        self.stop()
