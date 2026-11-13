"""Landmark normalisation, feature construction and augmentation.

Design decisions
----------------
* Wrist-relative: subtract landmark 0 -> translation invariant.
* Scale-invariant: divide by palm size (wrist -> middle-finger MCP).
* Handedness: slot 1 (Left) is mirrored (x -> -x) so every hand is expressed as a canonical
  right hand; one model therefore serves both hands.
* Rotation is deliberately NOT removed (orientation distinguishes some signs); instead the
  training set is augmented with small random rotations.
* Mirror augmentation flips x *without* swapping slots. It simulates MediaPipe handedness
  mislabelling, making the classifier robust to that failure mode.
* Dynamic features add a wrist trajectory (centred on the sequence mean and divided by palm
  size) so motion is camera-position and distance invariant. The Left-slot trajectory is mirrored
  as well, so a left-handed signer's circle looks like a right-handed signer's circle.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6
STATIC_DIM = 2 * 21 * 3 + 2        # 126 shape values + 2 presence flags = 128
DYNAMIC_DIM = STATIC_DIM + 4       # + wrist (x, y) for each hand = 132


def hand_scale(rel: np.ndarray) -> np.ndarray:
    """Palm size per hand, shape ``(..., 2)``; falls back to max landmark distance."""
    palm = np.linalg.norm(rel[..., 9, :], axis=-1)
    maxd = np.linalg.norm(rel, axis=-1).max(axis=-1)
    return np.maximum(np.where(palm > EPS, palm, maxd), EPS)


def canonical_mask(mask: np.ndarray) -> np.ndarray:
    """Presence mask after canonicalisation: a lone Left hand is moved into slot 0."""
    mask = np.asarray(mask, bool)
    only_left = (~mask[..., 0]) & mask[..., 1]
    return np.where(only_left[..., None], mask[..., ::-1], mask)


def _swap_single_left(x: np.ndarray, mask: np.ndarray, trailing: int) -> np.ndarray:
    """Swap the slot axis (located ``trailing`` axes from the end) where only the Left hand is present."""
    only_left = (~np.asarray(mask, bool)[..., 0]) & np.asarray(mask, bool)[..., 1]
    cond = only_left.reshape(only_left.shape + (1,) * trailing)
    return np.where(cond, np.flip(x, axis=-trailing), x)


def normalize_hands(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise ``raw (..., 2, 21, 3)`` -> same shape; absent hands become zeros.

    Steps: wrist-relative, palm-size scaling, mirror the Left slot to a canonical right hand,
    then move a lone Left hand into slot 0 so one-handed signs look identical for either hand.
    """
    raw = np.asarray(raw, np.float32)
    mask = np.asarray(mask, bool)
    rel = raw - raw[..., 0:1, :]
    norm = rel / hand_scale(rel)[..., None, None]
    norm[..., 1, :, 0] *= -1.0                       # mirror the Left slot to canonical right
    norm = np.where(mask[..., None, None], norm, 0.0).astype(np.float32)
    return _swap_single_left(norm, mask, 3)


def static_features(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Static feature vector ``(..., 128)`` from raw landmarks (handedness-invariant for one hand)."""
    norm = normalize_hands(raw, mask)
    flat = norm.reshape(norm.shape[:-3] + (126,))
    return np.concatenate([flat, canonical_mask(mask).astype(np.float32)], axis=-1)


def sequence_features(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dynamic features ``(..., T, 132)`` from ``raw (..., T, 2, 21, 3)``."""
    raw = np.asarray(raw, np.float32)
    mask = np.asarray(mask, bool)
    single = raw.ndim == 4
    if single:
        raw, mask = raw[None], mask[None]
    base = static_features(raw, mask)                              # (N, T, 128)
    wrist = raw[..., 0, :2]                                        # (N, T, 2, 2)
    rel = raw - raw[..., 0:1, :]
    palm = hand_scale(rel)                                         # (N, T, 2)
    m = mask.astype(np.float32)
    cnt = np.maximum(m.sum(axis=1), 1.0)                           # (N, 2)
    mean_w = (wrist * m[..., None]).sum(axis=1) / cnt[..., None]   # (N, 2, 2)
    mean_p = np.maximum((palm * m).sum(axis=1) / cnt, EPS)         # (N, 2)
    traj = (wrist - mean_w[:, None]) / mean_p[:, None, :, None]
    traj[..., 1, 0] *= -1.0                                        # mirror the Left slot trajectory too
    traj = traj * m[..., None]
    traj = _swap_single_left(traj, mask, 2)                        # lone Left hand -> slot 0
    out = np.concatenate([base, traj.reshape(raw.shape[0], raw.shape[1], 4)], axis=-1).astype(np.float32)
    return out[0] if single else out


def motion_score(raw: np.ndarray, mask: np.ndarray, smooth: int = 3) -> float:
    """Motion of a ``(T, 2, 21, 3)`` window: temporal spread of wrist position and fingertip extension.

    For each present hand we take 7 low-noise signals in palm units (wrist x, wrist y and the
    wrist-to-fingertip distance of the 5 fingers), smooth them with a ``smooth``-frame moving
    average and take the largest standard deviation over time. Landmark jitter is suppressed by
    the smoothing, whereas real signing (hand travel, finger opening/closing) is not. The larger
    value over the two hands is returned. Calibrated on synthetic data (see README).
    """
    raw = np.asarray(raw, np.float32)
    mask = np.asarray(mask, bool)
    rel = raw - raw[..., 0:1, :]
    palm = hand_scale(rel)                                            # (T, 2)
    tips = np.linalg.norm(rel[..., [4, 8, 12, 16, 20], :], axis=-1) / palm[..., None]   # (T, 2, 5)
    wrist = raw[..., 0, :2] / palm[..., None]                         # (T, 2, 2)
    sig = np.concatenate([wrist, tips], axis=-1)                      # (T, 2, 7)
    ker = np.ones(smooth) / smooth
    best = 0.0
    for h in range(2):
        x = sig[mask[:, h], h]
        if len(x) >= smooth + 2:
            sm = np.stack([np.convolve(x[:, j], ker, "valid") for j in range(x.shape[1])], axis=1)
            best = max(best, float(sm.std(axis=0).max()))
    return best


# ----------------------------------------------------------------------------- augmentation
def mirror_raw(raw: np.ndarray) -> np.ndarray:
    """Flip x (image-space) WITHOUT swapping hand slots (simulates handedness mislabelling)."""
    out = raw.copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def rotate_raw(raw: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Rotate in the image plane about the image centre; one angle per leading sample."""
    ang = np.asarray(angles_rad, np.float32).reshape((-1,) + (1,) * (raw.ndim - 2))
    c, s = np.cos(ang), np.sin(ang)
    x, y = raw[..., 0] - 0.5, raw[..., 1] - 0.5
    out = raw.copy()
    out[..., 0] = 0.5 + x * c - y * s
    out[..., 1] = 0.5 + x * s + y * c
    return out


def augment_raw(raw: np.ndarray, mask: np.ndarray, rng: np.random.Generator, copies: int = 3,
                noise_std: float = 0.004, rotation_deg: float = 15.0, scale_jitter: float = 0.1,
                mirror_prob: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Return ``copies`` augmented versions of every sample (original NOT included).

    Works for static ``(N, 2, 21, 3)`` and sequences ``(N, T, 2, 21, 3)``; the same random
    transform is applied to all frames of a sequence.
    """
    raw = np.tile(np.asarray(raw, np.float32), (copies,) + (1,) * (raw.ndim - 1))
    mask = np.tile(np.asarray(mask, bool), (copies,) + (1,) * (mask.ndim - 1))
    n = raw.shape[0]
    flip = rng.random(n) < mirror_prob
    if flip.any():
        raw[flip] = mirror_raw(raw[flip])
    raw = rotate_raw(raw, np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, n)))
    sc = (1.0 + rng.uniform(-scale_jitter, scale_jitter, n)).astype(np.float32)
    sc = sc.reshape((-1,) + (1,) * (raw.ndim - 1))
    raw = (0.5 + (raw - 0.5) * sc).astype(np.float32)
    raw += rng.normal(0.0, noise_std, raw.shape).astype(np.float32)
    return raw, mask
