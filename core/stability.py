"""Stability filter: commit a sign only after N consistent, confident frames."""
from __future__ import annotations

from typing import Optional


class StabilityFilter:
    """Commit a label after ``n_frames`` consecutive agreeing frames above ``threshold``.

    * Low-confidence frames and the ``unknown`` class are rejected (they break the streak).
    * After a commit, no further commit is possible for ``cooldown_s`` seconds.
    * With ``require_release`` the *same* sign cannot commit again until the signer released
      it (a different sign / unknown / no hand was seen) - this prevents "AAAA" while holding.
    """

    def __init__(self, n_frames: int = 8, threshold: float = 0.7, cooldown_s: float = 1.2,
                 unknown_label: str = "unknown", require_release: bool = True) -> None:
        self.n_frames = int(n_frames)
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.unknown_label = unknown_label
        self.require_release = require_release
        self._candidate: Optional[str] = None
        self._count = 0
        self._last_label: Optional[str] = None
        self._last_time = float("-inf")
        self._released = True

    @property
    def progress(self) -> tuple[Optional[str], float]:
        """(current candidate, fraction of the N frames collected)."""
        return self._candidate, min(1.0, self._count / max(1, self.n_frames))

    def reset(self) -> None:
        """Call when the hand disappears: clears the streak and counts as a release."""
        self._candidate, self._count, self._released = None, 0, True

    def update(self, label: Optional[str], confidence: float, now: float) -> Optional[str]:
        """Feed one frame; return the label if (and only if) it is committed on this frame."""
        effective = label if (label and label != self.unknown_label and confidence >= self.threshold) else None
        if effective is None:
            self._candidate, self._count, self._released = None, 0, True
            return None
        if effective != self._last_label:
            self._released = True
        if effective == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = effective, 1
        if self._count < self.n_frames:
            return None
        if now - self._last_time < self.cooldown_s:
            return None
        if self.require_release and effective == self._last_label and not self._released:
            return None
        self._last_label, self._last_time, self._released = effective, now, False
        self._count = 0
        return effective
