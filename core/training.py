"""Shared training helpers used by ``scripts/train_static.py`` and ``scripts/train_dynamic.py``."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.config import load_config
from core.dataset import LandmarkSet, cross_signer_split
from core.normalize import augment_raw, canonical_mask, normalize_hands, sequence_features, static_features

log = logging.getLogger(__name__)


def three_way_split(ds: LandmarkSet, seed: int, test_size: float, val_size: float):
    """Signer-disjoint train / val / test index arrays (+ split mode string)."""
    tr_all, te, mode = cross_signer_split(ds.y, ds.signer, test_size, seed)
    sub_tr, sub_val, _ = cross_signer_split(ds.y[tr_all], ds.signer[tr_all], val_size, seed + 1)
    return tr_all[sub_tr], tr_all[sub_val], te, mode


def make_features(ds: LandmarkSet, kind: str) -> np.ndarray:
    """Feature matrix for a landmark set."""
    return static_features(ds.raw, ds.mask) if kind == "static" else sequence_features(ds.raw, ds.mask)


def augmented_features(ds: LandmarkSet, idx: np.ndarray, kind: str, classes: list[str], seed: int):
    """Original + augmented features and integer labels for the given indices."""
    cfg = load_config()["training"]
    sub = ds.subset(idx)
    cls = {c: i for i, c in enumerate(classes)}
    y = np.array([cls[v] for v in sub.y])
    raw_a, mask_a = augment_raw(sub.raw, sub.mask, np.random.default_rng(seed), cfg["augment_copies"],
                                cfg["noise_std"], cfg["rotation_deg"], cfg["scale_jitter"])
    X = np.concatenate([make_features(sub, kind), make_features(
        LandmarkSet(raw_a, mask_a, sub.y, sub.signer), kind)])
    return X, np.concatenate([y, np.tile(y, cfg["augment_copies"])])


def build_mlp(in_dim: int, n_classes: int):
    """Small Keras MLP for static signs."""
    import keras
    from keras import layers
    m = keras.Sequential([layers.Input((in_dim,)), layers.Dense(128, activation="relu"), layers.Dropout(0.3),
                          layers.Dense(64, activation="relu"), layers.Dropout(0.2),
                          layers.Dense(n_classes, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_rnn(kind: str, T: int, d: int, n_classes: int, units: int):
    """LSTM or GRU classifier: identical architecture except for the recurrent cell."""
    import keras
    from keras import layers
    cell = layers.LSTM if kind == "lstm" else layers.GRU
    m = keras.Sequential([layers.Input((T, d)), cell(units), layers.Dropout(0.3),
                          layers.Dense(32, activation="relu"), layers.Dense(n_classes, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def compute_static_references(ds: LandmarkSet, classes: list[str], unknown: str = "unknown") -> dict:
    """Mean canonical hand per static class (used by practice mode and the two-way sign display)."""
    norm = normalize_hands(ds.raw, ds.mask)                       # (N, 2, 21, 3)
    present = canonical_mask(ds.mask)
    labels, refs = [], []
    for c in classes:
        if c == unknown:
            continue
        sel = ds.y == c
        hands = []
        for slot in (0, 1):
            s = sel & present[:, slot]
            if s.any():
                hands.append(norm[s, slot])
        if hands:
            labels.append(c)
            refs.append(np.concatenate(hands).mean(axis=0))
    return {"static_labels": np.array(labels), "static_refs": np.stack(refs).astype(np.float32)}


def compute_dynamic_prototypes(ds: LandmarkSet, classes: list[str]) -> dict:
    """One prototype sequence per class: the sample whose features are closest to the class mean."""
    feats = sequence_features(ds.raw, ds.mask)
    norm = normalize_hands(ds.raw, ds.mask)                       # (N, T, 2, 21, 3)
    labels, hands, trajs = [], [], []
    for c in classes:
        idx = np.where(ds.y == c)[0]
        if len(idx) == 0:
            continue
        mean = feats[idx].mean(axis=0)
        best = idx[np.argmin(((feats[idx] - mean) ** 2).sum(axis=(1, 2)))]
        slot = int(np.argmax(canonical_mask(ds.mask[best]).sum(axis=0)))
        labels.append(c)
        hands.append(norm[best, :, slot])
        trajs.append(feats[best][:, 128 + 2 * slot:130 + 2 * slot])
    return {"dynamic_labels": np.array(labels), "dynamic_protos": np.stack(hands).astype(np.float32),
            "dynamic_trajs": np.stack(trajs).astype(np.float32)}
