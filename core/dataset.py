"""Landmark dataset container (``.npz``) and cross-signer splitting."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


@dataclass
class LandmarkSet:
    """Raw landmarks + labels + signer ids for one language/kind."""
    raw: np.ndarray            # (N, 2, 21, 3) static  or  (N, T, 2, 21, 3) dynamic
    mask: np.ndarray           # (N, 2)               or  (N, T, 2)
    y: np.ndarray              # (N,) str labels
    signer: np.ndarray         # (N,) str signer ids ("unknown" when not available)
    lang: str = ""
    kind: str = "static"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "LandmarkSet":
        return LandmarkSet(self.raw[idx], self.mask[idx], self.y[idx], self.signer[idx], self.lang, self.kind, self.meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, raw=self.raw, mask=self.mask, y=self.y.astype(str), signer=self.signer.astype(str),
                            lang=self.lang, kind=self.kind, meta=json.dumps(self.meta))

    @staticmethod
    def load(path: str | Path) -> "LandmarkSet":
        z = np.load(path, allow_pickle=False)
        return LandmarkSet(z["raw"], z["mask"], z["y"], z["signer"], str(z["lang"]), str(z["kind"]),
                           json.loads(str(z["meta"])))


def concat(sets: list[LandmarkSet]) -> LandmarkSet:
    """Concatenate several sets of the same kind/language."""
    if not sets:
        raise ValueError("no datasets to concatenate")
    first = sets[0]
    return LandmarkSet(np.concatenate([s.raw for s in sets]), np.concatenate([s.mask for s in sets]),
                       np.concatenate([s.y for s in sets]), np.concatenate([s.signer for s in sets]),
                       first.lang, first.kind, {"sources": [s.meta for s in sets]})


def load_many(paths: list[str | Path]) -> LandmarkSet:
    """Load and concatenate ``.npz`` landmark files."""
    return concat([LandmarkSet.load(p) for p in paths])


def cross_signer_split(y: np.ndarray, signer: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Split so that test signers never appear in training.

    Returns ``(train_idx, test_idx, mode)``. If there are fewer than 3 distinct signers the
    function falls back to a stratified random split and returns mode ``"random_NOT_cross_signer"``
    so that reports can state honestly that the evaluation is not signer-independent.
    """
    groups = np.asarray(signer).astype(str)
    if len(set(groups)) >= 3:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(np.zeros(len(y)), y, groups))
        return tr, te, "cross_signer"
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(np.zeros(len(y)), y))
    return tr, te, "random_NOT_cross_signer"
