"""Speech-to-text for the hearing person's reply (two-way mode).

Audio comes from Streamlit's ``st.audio_input`` as WAV bytes, so NO PyAudio / microphone driver
is required. Backends (``stt.backend`` in config.yaml):

* ``google``  - SpeechRecognition's free Google Web Speech endpoint (needs internet; default).
* ``whisper`` - offline, needs ``pip install faster-whisper`` (optional, not installed by default).
* ``auto``    - whisper if installed, otherwise google.
"""
from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass

from core.config import load_config

log = logging.getLogger(__name__)


@dataclass
class STTResult:
    text: str
    backend: str
    error: str = ""


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe_wav(wav_bytes: bytes, lang: str = "en") -> STTResult:
    """Transcribe WAV bytes. ``lang`` is ``"en"`` or ``"ur"``. Never raises."""
    cfg = load_config()["stt"]
    backend = cfg["backend"]
    if backend == "auto":
        backend = "whisper" if whisper_available() else "google"
    try:
        if backend == "whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
                fh.write(wav_bytes)
                fh.flush()
                segments, _ = model.transcribe(fh.name, language=lang)
                return STTResult(" ".join(s.text.strip() for s in segments).strip(), "whisper")
        import speech_recognition as sr
        rec = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = rec.record(src)
        text = rec.recognize_google(audio, language=cfg["languages"].get(lang, "en-US"))
        return STTResult(text, "google")
    except Exception as exc:
        name = type(exc).__name__
        hint = ("No speech understood." if name == "UnknownValueError" else
                "Speech service unreachable (internet needed for the Google backend; or install faster-whisper)."
                if name in ("RequestError", "URLError", "ConnectionError") else f"{name}: {exc}")
        log.warning("STT failed: %s", hint)
        return STTResult("", backend, hint)
