"""Model loading/saving and a uniform prediction interface.

File layout inside a models directory (``models/trained`` or ``models/demo``)::

    static_<LANG>_rf.joblib   static_<LANG>_mlp.keras   static_<LANG>_labels.json   static_<LANG>_meta.json
    dynamic_<LANG>_lstm.keras dynamic_<LANG>_gru.keras  dynamic_<LANG>_labels.json  dynamic_<LANG>_meta.json
    references_<LANG>.npz
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import path_of

log = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


class Predictor:
    """Wraps a sklearn or Keras classifier behind ``predict_proba(x) -> (K,) probabilities``."""

    def __init__(self, model, kind: str, name: str, classes: list[str]) -> None:
        self.model, self.kind, self.name, self.classes = model, kind, name, classes
        self._lock = threading.Lock()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilities for ONE sample (static ``(128,)`` or sequence ``(T, 132)``)."""
        batch = np.asarray(x, np.float32)[None]
        with self._lock:
            if self.kind == "sklearn":
                p = self.model.predict_proba(batch)[0]
                full = np.zeros(len(self.classes), np.float32)
                full[np.asarray(self.model.classes_, int)] = p
                return full
            return np.asarray(self.model(batch, training=False))[0].astype(np.float32)

    def predict_proba_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        with self._lock:
            if self.kind == "sklearn":
                p = self.model.predict_proba(x)
                full = np.zeros((len(x), len(self.classes)), np.float32)
                full[:, np.asarray(self.model.classes_, int)] = p
                return full
            return np.asarray(self.model.predict(x, verbose=0), np.float32)


@dataclass
class Component:
    """One loaded (language, kind) component, e.g. the ASL static classifier."""
    kind: str                                   # "static" | "dynamic"
    classes: list[str]
    predictors: dict[str, Predictor]            # e.g. {"rf": ..., "mlp": ...}
    default: str
    meta: dict
    source_dir: Path

    @property
    def is_demo(self) -> bool:
        return bool(self.meta.get("is_demo", False))

    def predict(self, x: np.ndarray, which: Optional[str] = None) -> np.ndarray:
        return self.predictors[which or self.default].predict_proba(x)


@dataclass
class LanguageModels:
    """All components for one language, plus practice references."""
    lang: str
    static: Optional[Component] = None
    dynamic: Optional[Component] = None
    references: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def is_demo(self) -> bool:
        comps = [c for c in (self.static, self.dynamic) if c]
        return bool(comps) and any(c.is_demo for c in comps)


def _load_component(kind: str, lang: str, dirs: list[Path]) -> tuple[Optional[Component], list[str]]:
    problems: list[str] = []
    for d in dirs:
        meta_p, lab_p = d / f"{kind}_{lang}_meta.json", d / f"{kind}_{lang}_labels.json"
        if not (meta_p.exists() and lab_p.exists()):
            continue
        meta, classes = load_json(meta_p), load_json(lab_p)["classes"]
        preds: dict[str, Predictor] = {}
        for name in meta.get("models", []):
            try:
                if name == "rf":
                    import joblib
                    preds[name] = Predictor(joblib.load(d / f"{kind}_{lang}_rf.joblib"), "sklearn", name, classes)
                else:
                    from core.compat import ensure_import_order
                    ensure_import_order(webrtc=False)   # never import webrtc late: it segfaults after tensorflow
                    import keras
                    preds[name] = Predictor(keras.models.load_model(d / f"{kind}_{lang}_{name}.keras", compile=False),
                                            "keras", name, classes)
            except Exception as exc:
                problems.append(f"{kind}/{lang}/{name} in {d.name}: {exc}")
                log.warning("Failed to load %s %s %s: %s", kind, lang, name, exc)
        if preds:
            default = meta.get("default") if meta.get("default") in preds else next(iter(preds))
            return Component(kind, classes, preds, default, meta, d), problems
    return None, problems


def load_language_models(lang: str, dirs: Optional[list[Path]] = None) -> LanguageModels:
    """Load models for ``lang``; trained models take precedence over demo models per component."""
    dirs = dirs or [path_of("models_trained"), path_of("models_demo")]
    lm = LanguageModels(lang)
    for kind in ("static", "dynamic"):
        comp, probs = _load_component(kind, lang, dirs)
        lm.problems += probs
        setattr(lm, kind, comp)
        if comp is None:
            lm.problems.append(f"No {kind} model for {lang}. Run: python scripts/build_demo.py  (demo)  "
                               f"or scripts/train_{kind}.py (real data).")
    for d in dirs:
        ref = d / f"references_{lang}.npz"
        if ref.exists():
            z = np.load(ref, allow_pickle=False)
            lm.references = {k: z[k] for k in z.files}
            break
    return lm
