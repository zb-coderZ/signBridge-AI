"""Offline text-to-speech with pyttsx3, run on a worker thread so video never blocks."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from core.config import load_config

log = logging.getLogger(__name__)


class TTSEngine:
    """Non-blocking speaker. ``speak()`` enqueues text and returns a status string immediately.

    pyttsx3 is initialised *inside* the worker thread (its drivers are not thread-safe). If no
    driver is available (e.g. Linux without espeak) the engine reports ``available=False`` and
    every call degrades to a no-op. If Urdu is requested but no Urdu voice is installed, the
    default voice is used and the status says so.
    """

    def __init__(self) -> None:
        cfg = load_config()["tts"]
        self._rate = int(cfg["rate"])
        self._urdu_kw = [k.lower() for k in cfg["urdu_keywords"]]
        self._q: queue.Queue[Optional[tuple[str, str]]] = queue.Queue(maxsize=20)
        self.available: Optional[bool] = None       # None = not yet initialised
        self.error = ""
        self.voices: list[tuple[str, str, str]] = []  # (id, name, languages)
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()

    # ---- voice selection
    def _find_voice(self, lang: str) -> Optional[str]:
        kws = self._urdu_kw if lang == "ur" else ["english", "en-us", "en_us", "en-gb", "en_gb"]
        for vid, name, langs in self.voices:
            blob = f"{vid} {name} {langs}".lower()
            if any(k in blob for k in kws):
                return vid
        return None

    def has_urdu_voice(self) -> bool:
        self._ready.wait(5)
        return bool(self.available) and self._find_voice("ur") is not None

    def _worker(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            for v in engine.getProperty("voices") or []:
                self.voices.append((str(v.id), str(getattr(v, "name", "")), str(getattr(v, "languages", ""))))
            self.available = True
        except Exception as exc:
            self.available, self.error = False, f"{type(exc).__name__}: {exc}"
            log.warning("TTS unavailable: %s", self.error)
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._q.get()
            if item is None:
                break
            text, lang = item
            try:
                vid = self._find_voice(lang)
                if vid:
                    engine.setProperty("voice", vid)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                log.warning("TTS failed: %s", exc)

    def speak(self, text: str, lang: str = "en") -> str:
        """Queue ``text`` for speech. ``lang`` is ``"en"`` or ``"ur"``. Returns a short status message."""
        text = text.strip()
        if not text:
            return "Nothing to speak."
        self._ready.wait(5)
        if not self.available:
            return f"Text-to-speech unavailable on this machine ({self.error or 'no driver'})."
        note = ""
        if lang == "ur" and self._find_voice("ur") is None:
            note = " No Urdu voice is installed - using the default voice (install one in your OS language settings)."
        try:
            self._q.put_nowait((text, lang))
        except queue.Full:
            return "Speech queue is full; skipped."
        return "Speaking." + note

    def shutdown(self) -> None:
        self._q.put(None)
