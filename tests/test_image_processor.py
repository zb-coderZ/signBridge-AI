import time

import av
import numpy as np

from app.components.image_processor import ImageVideoProcessor
from core.image_engine import ImageResult


class FakeEngine:
    def __init__(self) -> None:
        self.last = ImageResult()
        self.calls = 0

    def process(self, image, timestamp_ms=None, model_name="random_forest"):
        self.calls += 1
        return self.last


def test_webcam_recv_returns_without_waiting_for_inference():
    engine = FakeEngine()
    processor = ImageVideoProcessor(engine)
    frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24")
    started = time.perf_counter()
    for _ in range(5):
        processor.recv(frame)
    elapsed = time.perf_counter() - started
    time.sleep(0.05)
    processor.stop()

    assert elapsed < 0.2
    assert engine.calls >= 1
