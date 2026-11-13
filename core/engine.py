"""Real-time recognition engine (camera-agnostic and Streamlit-agnostic, therefore unit-testable).

Per frame:  detect hands -> slot arrays -> features -> [static | dynamic] model(s) ->
language router -> stability filter -> sentence builder + history (+ optional practice score).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from core.config import load_config
from core.hand_detector import DetectorUnavailable, HandDetector, draw_hands
from core.landmarks import observations_to_slots, resample_sequence
from core.language_router import LanguageRouter
from core.models_io import LanguageModels
from core.normalize import canonical_mask, motion_score, normalize_hands, sequence_features, static_features
from core.practice import PracticeResult, similarity
from core.sentence_builder import SentenceBuilder
from core.stability import StabilityFilter

log = logging.getLogger(__name__)


@dataclass
class Settings:
    """User-adjustable settings (sidebar)."""
    language_mode: str = "Auto"          # Auto | PSL | ASL
    sign_mode: str = "Auto"              # Auto | Static only | Dynamic only
    threshold: float = 0.70
    dynamic_threshold: float = 0.75
    stable_frames: int = 8
    cooldown_s: float = 1.2
    motion_threshold: float = 0.14
    show_skeleton: bool = True


@dataclass
class Prediction:
    """One model decision for the active language."""
    language: str
    kind: str                            # "static" | "dynamic"
    label: str
    confidence: float
    top: list[tuple[str, float]]
    per_language_conf: dict[str, float]


@dataclass
class FrameResult:
    """What the UI needs after one frame."""
    hands: int = 0
    prediction: Optional[Prediction] = None
    committed: Optional[str] = None
    motion: float = 0.0
    kind: str = "static"
    language: str = ""
    progress: float = 0.0
    practice: Optional[PracticeResult] = None
    note: str = ""


class RecognitionEngine:
    """Owns all per-session recognition state. Thread-safe (one lock)."""

    def __init__(self, models: dict[str, LanguageModels], detector_mode: str = "video") -> None:
        cfg = load_config()
        rc = cfg["recognition"]
        self.cfg, self.rc = cfg, rc
        self.models = models
        self.languages = tuple(cfg["languages"])
        self.T = int(cfg["sequence_length"])
        self.settings = Settings(threshold=rc["confidence_threshold"], dynamic_threshold=rc["dynamic_confidence_threshold"],
                                 stable_frames=rc["stable_frames"], cooldown_s=rc["cooldown_s"],
                                 motion_threshold=rc["motion_threshold"])
        self.lock = threading.RLock()
        self.router = LanguageRouter(self.languages, rc["auto_window_s"])
        self.stability = StabilityFilter(rc["stable_frames"], rc["confidence_threshold"], rc["cooldown_s"],
                                         rc["unknown_label"], rc["require_release"])
        self.sentence = SentenceBuilder()
        self.history: list[dict] = []
        self.conf_history: deque = deque(maxlen=300)
        self.on_commit: list[Callable[[str, str, float], None]] = []
        self.last: FrameResult = FrameResult()
        self.practice_target: Optional[tuple[str, str]] = None
        self._buf: deque = deque(maxlen=120)           # (t, raw, mask)
        self._no_hand = 0
        self._frame_i = 0
        self._dyn_until = -1.0
        self._dyn_cache: dict[str, tuple[np.ndarray, str]] = {}
        self._detector_mode = detector_mode
        self._detector: Optional[HandDetector] = None
        self.detector_error = ""
        self._detector_failed_at = -1e9
        self.fps = 0.0
        self.latency_ms = 0.0
        self._t_prev: Optional[float] = None

    # ------------------------------------------------------------------ settings
    def apply_settings(self, s: Settings) -> None:
        """Push sidebar settings into the live components."""
        with self.lock:
            self.settings = s
            self.router.set_mode(s.language_mode)
            self.stability.n_frames = int(s.stable_frames)
            self.stability.cooldown_s = float(s.cooldown_s)

    @property
    def has_models(self) -> bool:
        return any(m.static or m.dynamic for m in self.models.values())

    # ------------------------------------------------------------------ core pipeline
    def _predict_language(self, lang: str, kind: str, feats_static: np.ndarray, now: float) -> Optional[Prediction]:
        lm = self.models.get(lang)
        comp = None if lm is None else getattr(lm, kind)
        if comp is None:
            return None
        if kind == "static":
            probs = comp.predict(feats_static)
        else:                                               # dynamic: every n-th frame, else reuse the cache
            cached = self._dyn_cache.get(lang)
            if cached is not None and self._frame_i % int(self.rc["dynamic_every_n_frames"]) != 0:
                probs = cached[0]
            else:
                win = [(r, m) for t, r, m in self._buf if now - t <= self.rc["sequence_window_s"]]
                if len(win) < 8:
                    return None
                raw, mask = resample_sequence(win, self.T)
                probs = comp.predict(sequence_features(raw, mask))
                self._dyn_cache[lang] = (probs, "")
        order = np.argsort(-probs)
        top = [(comp.classes[i], float(probs[i])) for i in order[: int(self.rc["top_k"])]]
        return Prediction(lang, kind, top[0][0], top[0][1], top, {})

    def process_landmarks(self, raw: np.ndarray, mask: np.ndarray, now: Optional[float] = None) -> FrameResult:
        """Run the full pipeline on one frame of slot arrays (``raw (2,21,3)``, ``mask (2,)``)."""
        now = time.monotonic() if now is None else now
        with self.lock:
            t0 = time.perf_counter()
            self._frame_i += 1
            res = FrameResult(hands=int(mask.sum()))
            st = self.settings
            if not mask.any():
                self._no_hand += 1
                if self._no_hand > 5:
                    self._buf.clear(); self._dyn_cache.clear()
                self.stability.reset()
                res.language = self.router.active(now)
                self.last = res
                return res
            self._no_hand = 0
            self._buf.append((now, raw.copy(), mask.copy()))

            motion = 0.0
            win = list(self._buf)[-int(self.rc["motion_window"]):]
            if len(win) >= int(self.rc["motion_window"]):
                motion = motion_score(np.stack([w[1] for w in win]), np.stack([w[2] for w in win]))
            res.motion = motion
            if st.sign_mode == "Dynamic only":
                kind = "dynamic"
            elif st.sign_mode == "Static only":
                kind = "static"
            else:
                if motion > st.motion_threshold:
                    self._dyn_until = now + self.rc["dynamic_hold_s"]
                kind = "dynamic" if now <= self._dyn_until else "static"
            res.kind = kind

            feats = static_features(raw, mask)
            langs = list(self.languages) if self.router.mode == "Auto" else [self.router.mode]
            preds: dict[str, Prediction] = {}
            for lang in langs:
                p = self._predict_language(lang, kind, feats, now)
                if p is not None:
                    preds[lang] = p
                    self.router.update(lang, p.confidence, now)
            active = self.router.active(now)
            res.language = active
            pred = preds.get(active) or (next(iter(preds.values())) if preds else None)
            if pred is not None:
                pred.per_language_conf = {l: p.confidence for l, p in preds.items()}
                res.prediction = pred
                thr = st.dynamic_threshold if kind == "dynamic" else st.threshold
                self.stability.threshold = thr
                # Auto mode: do not commit a static letter until the motion window is full, otherwise the
                # first frames of a moving sign would be committed as a static letter.
                warming = st.sign_mode == "Auto" and len(win) < int(self.rc["motion_window"])
                committed = None if warming else self.stability.update(pred.label, pred.confidence, now)
                res.progress = 0.0 if warming else self.stability.progress[1]
                self.conf_history.append((now, pred.confidence, pred.label))
                if committed:
                    res.committed = committed
                    self._commit(committed, pred, now)
            else:
                self.stability.reset()
                res.note = "No model available for this language/sign type."

            if self.practice_target:
                res.practice = self._practice(raw, mask)
            self.latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last = res
            return res

    def _commit(self, label: str, pred: Prediction, now: float) -> None:
        self.sentence.add(label)
        self.history.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "language": pred.language,
                             "type": pred.kind, "sign": label, "confidence": round(pred.confidence, 3)})
        del self.history[: -int(self.rc["history_len"])]
        for cb in self.on_commit:
            try:
                cb(label, pred.language, pred.confidence)
            except Exception:
                log.exception("on_commit callback failed")

    def _practice(self, raw: np.ndarray, mask: np.ndarray) -> Optional[PracticeResult]:
        lang, label = self.practice_target  # type: ignore[misc]
        refs = self.models.get(lang).references if self.models.get(lang) else {}
        labels = [str(l) for l in refs.get("static_labels", [])]
        if label not in labels:
            return None
        ref = refs["static_refs"][labels.index(label)]
        norm = normalize_hands(raw, mask)
        slot = 0 if canonical_mask(mask)[0] else 1
        return similarity(norm[slot], ref)

    def clear_all(self) -> None:
        """Reset sentence, history and filters."""
        with self.lock:
            self.sentence.clear(); self.history.clear(); self.conf_history.clear()
            self.stability.reset(); self._buf.clear(); self.router.reset()

    # ------------------------------------------------------------------ camera frames
    def _ensure_detector(self) -> Optional[HandDetector]:
        if self._detector is not None:
            return self._detector
        if time.monotonic() - self._detector_failed_at < 15:       # do not hammer the network every frame
            return None
        try:
            self._detector = HandDetector(self._detector_mode)
            self.detector_error = ""
        except DetectorUnavailable as exc:
            self.detector_error = str(exc)
            self._detector_failed_at = time.monotonic()
        except Exception as exc:                                   # e.g. mediapipe missing
            self.detector_error = f"{type(exc).__name__}: {exc}"
            self._detector_failed_at = time.monotonic()
        return self._detector

    def process_frame(self, bgr: np.ndarray, now: Optional[float] = None) -> tuple[np.ndarray, FrameResult]:
        """Detect hands in a BGR frame, run the pipeline and return ``(annotated_bgr, result)``."""
        t0 = time.perf_counter()
        if self.cfg["hand_detector"]["mirror_input"]:
            bgr = cv2.flip(bgr, 1)
        det = self._ensure_detector()
        if det is None:
            cv2.putText(bgr, "Hand model unavailable - see app status", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return bgr, FrameResult(note=self.detector_error)
        hands = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(time.monotonic() * 1000))
        if self.settings.show_skeleton:
            draw_hands(bgr, hands)
        raw, mask = observations_to_slots(hands)
        res = self.process_landmarks(raw, mask, now)
        self._overlay(bgr, res)
        dt_s = time.perf_counter() - t0
        self.fps = 0.9 * self.fps + 0.1 / max(dt_s, 1e-6) if self.fps else 1.0 / max(dt_s, 1e-6)
        return bgr, res

    def _overlay(self, img: np.ndarray, res: FrameResult) -> None:
        if res.prediction:
            p = res.prediction
            txt = f"{p.language} {p.kind[:3]}: {p.label} {p.confidence * 100:.0f}%"
            cv2.rectangle(img, (0, 0), (img.shape[1], 44), (30, 30, 30), -1)
            cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (0, 44), (int(img.shape[1] * res.progress), 50), (0, 200, 120), -1)
        text = self.sentence.text
        if text:
            cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (30, 30, 30), -1)
            cv2.putText(img, text[-48:], (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ UI snapshot
    def snapshot(self) -> dict:
        """Thread-safe copy of everything the dashboard shows."""
        with self.lock:
            return {"last": self.last, "text": self.sentence.text, "history": list(self.history),
                    "conf": list(self.conf_history), "fps": self.fps, "latency_ms": self.latency_ms,
                    "detector_error": self.detector_error, "avg_conf": self.router.averages(time.monotonic())}
