"""streamlit-webrtc video processor that feeds frames into a :class:`RecognitionEngine`."""
from __future__ import annotations

import logging

import av
from streamlit_webrtc import VideoProcessorBase

from core.engine import RecognitionEngine

log = logging.getLogger(__name__)


class SignProcessor(VideoProcessorBase):
    """Runs the engine on every frame; never lets an exception kill the video stream."""

    def __init__(self, engine: RecognitionEngine | None = None) -> None:
        self.engine = engine

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            if self.engine is not None:
                img, _ = self.engine.process_frame(img)
        except Exception:
            log.exception("frame processing failed")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
