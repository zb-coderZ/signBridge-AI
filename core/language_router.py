"""Language selection: manual override or Auto-detect by average confidence over a sliding window."""
from __future__ import annotations

from collections import deque
from typing import Optional


class LanguageRouter:
    """Chooses which language's prediction is shown.

    * ``mode`` = ``"PSL"`` / ``"ASL"`` -> always that language (manual selection overrides Auto).
    * ``mode`` = ``"Auto"``            -> language whose mean top-confidence over the last
      ``window_s`` seconds is highest. A small hysteresis margin prevents flicker.
    """

    def __init__(self, languages: tuple[str, ...] = ("PSL", "ASL"), window_s: float = 3.0, margin: float = 0.03) -> None:
        self.languages = tuple(languages)
        self.window_s, self.margin = window_s, margin
        self.mode = "Auto"
        self._hist: dict[str, deque[tuple[float, float]]] = {l: deque() for l in self.languages}
        self._current: str = self.languages[0]

    def set_mode(self, mode: str) -> None:
        """Set ``"Auto"`` or a language code."""
        self.mode = mode if mode == "Auto" or mode in self.languages else "Auto"

    def update(self, lang: str, confidence: float, now: float) -> None:
        """Record the top confidence of ``lang`` for the current frame."""
        if lang in self._hist:
            self._hist[lang].append((now, float(confidence)))

    def reset(self) -> None:
        for q in self._hist.values():
            q.clear()

    def averages(self, now: float) -> dict[str, Optional[float]]:
        """Mean confidence per language over the window (None if no data)."""
        out: dict[str, Optional[float]] = {}
        for lang, q in self._hist.items():
            while q and now - q[0][0] > self.window_s:
                q.popleft()
            out[lang] = sum(c for _, c in q) / len(q) if q else None
        return out

    def active(self, now: float) -> str:
        """Language to use right now."""
        if self.mode != "Auto":
            self._current = self.mode
            return self._current
        avg = self.averages(now)
        valid = {l: a for l, a in avg.items() if a is not None}
        if valid:
            best = max(valid, key=valid.get)
            cur = valid.get(self._current)
            if cur is None or valid[best] > cur + self.margin:
                self._current = best
        return self._current
