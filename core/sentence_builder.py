"""Sentence builder (letters + words) and gloss-to-sentence clean-up."""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)


class SentenceBuilder:
    """Accumulates committed signs.

    Single letters are concatenated into words; multi-character signs (words) are separated
    by spaces automatically. ``space``, ``backspace`` and ``clear`` are supported.
    """

    def __init__(self) -> None:
        self._units: list[tuple[str, bool]] = []   # (text, auto_inserted)

    @staticmethod
    def _is_word(token: str) -> bool:
        return len(token) > 1 and token.strip() != ""

    def _last_text(self) -> Optional[str]:
        return self._units[-1][0] if self._units else None

    def add(self, token: str) -> None:
        """Append a committed sign."""
        token = token.strip()
        if not token:
            return
        last = self._last_text()
        if last is not None and last != " ":
            if self._is_word(token) or self._is_word(last):
                self._units.append((" ", True))
        self._units.append((token, False))

    def space(self) -> None:
        """Insert one explicit space (never two in a row, never leading)."""
        if self._units and self._last_text() != " ":
            self._units.append((" ", False))

    def backspace(self) -> None:
        """Remove the last committed unit (and an automatic space preceding it)."""
        if self._units:
            self._units.pop()
            if self._units and self._units[-1] == (" ", True):
                self._units.pop()

    def clear(self) -> None:
        """Remove everything."""
        self._units.clear()

    @property
    def text(self) -> str:
        """Current text."""
        return "".join(t for t, _ in self._units)

    def __len__(self) -> int:
        return len(self._units)


# --------------------------------------------------------------------------- gloss clean-up
_PRONOUNS = {"ME": "I", "MY": "my", "MINE": "mine", "YOU": "you", "YOUR": "your"}
_QUESTION = {"WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN"}


def rule_based_sentence(text: str) -> str:
    """Cheap, dependency-free gloss clean-up: de-duplicate, map pronouns, capitalise, punctuate."""
    words = [w for w in text.replace("_", " ").split() if w]
    cleaned: list[str] = []
    for w in words:
        if cleaned and cleaned[-1].upper() == w.upper():
            continue
        cleaned.append(w)
    if not cleaned:
        return ""
    is_q = any(w.upper() in _QUESTION for w in cleaned)
    out = [_PRONOUNS.get(w.upper(), w.lower() if w.isupper() and len(w) > 1 else w) for w in cleaned]
    sentence = " ".join(out)
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".?!":
        sentence += "?" if is_q else "."
    return sentence


def llm_sentence(text: str, cfg: dict) -> Optional[str]:
    """Optional LLM polish via the Anthropic HTTP API. Returns None if unavailable/failed."""
    key = os.environ.get(cfg["llm"]["env_key"], "")
    if not key or not text.strip():
        return None
    body = json.dumps({
        "model": cfg["llm"]["model"], "max_tokens": cfg["llm"]["max_tokens"],
        "messages": [{"role": "user", "content":
                      "Convert this sign-language gloss into one natural English sentence. "
                      f"Reply with only the sentence.\nGloss: {text}"}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "x-api-key": key,
                                          "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return "".join(b.get("text", "") for b in data.get("content", [])).strip() or None
    except Exception as exc:  # network / auth / quota: never fatal
        log.warning("LLM clean-up failed (%s); using rule-based fallback", exc)
        return None


def gloss_to_sentence(text: str, cfg: dict, use_llm: bool = False) -> tuple[str, str]:
    """Return ``(sentence, method)`` where method is ``"llm"`` or ``"rules"``."""
    if use_llm:
        out = llm_sentence(text, cfg)
        if out:
            return out, "llm"
    return rule_based_sentence(text), "rules"
