"""Synthetic hand-landmark generator used ONLY for the demo model and for tests.

It builds a parametric 21-point hand skeleton from finger-curl parameters and adds
per-"signer" variation (hand size, finger proportions, curl bias, rotation, handedness).

IMPORTANT: the poses below are coarse finger-extension patterns. They are NOT accurate
ASL or PSL handshapes and exist solely so the whole pipeline can be demonstrated and
tested without any dataset. Never report their accuracy as sign-language accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# pose vector: [thumb_out, curl_index, curl_middle, curl_ring, curl_pinky, spread]  (curl 0=extended, 1=curled)
POSES_ASL: dict[str, list[float]] = {
    "A": [0.3, 1, 1, 1, 1, 0.0],
    "B": [0.0, 0, 0, 0, 0, 0.0],
    "I": [0.0, 1, 1, 1, 0, 0.0],
    "L": [1.0, 0, 1, 1, 1, 0.0],
    "V": [0.0, 0, 0, 1, 1, 0.35],
    "W": [0.0, 0, 0, 0, 1, 0.25],
    "Y": [1.0, 1, 1, 1, 0, 0.0],
}
POSES_PSL: dict[str, list[float]] = {     # neutral ids - these are NOT real PSL letters
    "P1": [0.0, 0, 1, 1, 1, 0.0],
    "P2": [0.0, 0, 1, 1, 0, 0.0],
    "P3": [1.0, 0, 0, 1, 1, 0.3],
    "P4": [1.0, 0, 0, 0, 0, 0.9],
    "P5": [1.0, 0, 1, 1, 0, 0.0],
    "P6": [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
}
STATIC_POSES = {"ASL": POSES_ASL, "PSL": POSES_PSL}
DYNAMIC_LABELS = ["WAVE", "CIRCLE", "NOD", "SWIPE", "PINCH"]   # motion patterns, not real signs
UNKNOWN = "unknown"

_MCP = np.array([[-0.28, -0.95], [0.0, -1.0], [0.26, -0.95], [0.50, -0.82]])
_SEG = np.array([[0.55, 0.35, 0.30], [0.60, 0.38, 0.32], [0.55, 0.35, 0.30], [0.42, 0.27, 0.25]])
_THUMB_SEG = np.array([0.45, 0.35, 0.30])
_THUMB_BASE = np.array([-0.25, -0.25])
_BASE_ANGLE = np.deg2rad([-6.0, 0.0, 6.0, 14.0])
_SPREAD_FACTOR = np.array([-1.0, 0.5, 1.2, 2.0])
_BEND = np.deg2rad([70.0, 95.0, 65.0])


@dataclass
class SignerStyle:
    """Fixed per-signer variation."""
    scale: float = 0.16
    seg_mult: np.ndarray = field(default_factory=lambda: np.ones(5))
    curl_bias: np.ndarray = field(default_factory=lambda: np.zeros(5))
    rot_deg: float = 0.0
    centre: tuple[float, float] = (0.5, 0.55)
    left_handed: bool = False


def make_signers(n: int, rng: np.random.Generator) -> list[SignerStyle]:
    """Create ``n`` distinct virtual signers (every 3rd is left-handed to exercise mirroring)."""
    out = []
    for i in range(n):
        out.append(SignerStyle(scale=float(rng.uniform(0.12, 0.22)),
                               seg_mult=rng.normal(1.0, 0.06, 5),
                               curl_bias=rng.normal(0.0, 0.06, 5),
                               rot_deg=float(rng.normal(0.0, 6.0)),
                               centre=(float(rng.uniform(0.4, 0.6)), float(rng.uniform(0.5, 0.65))),
                               left_handed=(i % 3 == 2)))
    return out


def _rot_x(d: np.ndarray, a: float) -> np.ndarray:
    return np.array([d[0], d[1] * np.cos(a) - d[2] * np.sin(a), d[1] * np.sin(a) + d[2] * np.cos(a)])


def hand_from_pose(pose: np.ndarray, style: SignerStyle, rng: np.random.Generator,
                   centre: tuple[float, float] | None = None, rot_extra_deg: float = 0.0,
                   noise: float = 0.004, pose_noise: float = 0.07) -> np.ndarray:
    """Build one right-handed ``(21, 3)`` hand in normalised image coordinates."""
    p = np.asarray(pose, float).copy()
    p[:5] = np.clip(p[:5] + rng.normal(0, pose_noise, 5) + style.curl_bias, 0, 1)
    p[5] = max(0.0, p[5] + rng.normal(0, 0.03))
    thumb_out, curls, spread = p[0], p[1:5], p[5]
    pts = np.zeros((21, 3))
    # four fingers
    for f in range(4):
        pos = np.array([_MCP[f, 0], _MCP[f, 1], 0.0])
        ang = _BASE_ANGLE[f] + np.deg2rad(spread * 40.0) * _SPREAD_FACTOR[f]
        d0 = np.array([np.sin(ang), -np.cos(ang), 0.0])
        base = 5 + 4 * f
        pts[base] = pos
        cum = 0.0
        for j in range(3):
            cum += _BEND[j] * curls[f]
            pos = pos + _rot_x(d0, cum) * _SEG[f, j] * style.seg_mult[f + 1]
            pts[base + 1 + j] = pos
    # thumb: pointing left-up when out, across the palm when tucked
    th = np.deg2rad(-55.0 * thumb_out + 70.0 * (1 - thumb_out))
    d0 = np.array([np.sin(th), -np.cos(th), 0.0])
    pos = np.array([_THUMB_BASE[0], _THUMB_BASE[1], 0.0])
    pts[1] = pos
    cum = 0.0
    for j in range(3):
        cum += np.deg2rad(25.0) * (1 - thumb_out)
        pos = pos + _rot_x(d0, cum) * _THUMB_SEG[j] * style.seg_mult[0]
        pts[2 + j] = pos
    # global transform: rotation, scale, translation (y is image-down)
    a = np.deg2rad(style.rot_deg + rot_extra_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    pts = (pts @ R.T) * style.scale
    cx, cy = centre if centre is not None else style.centre
    pts[:, 0] += cx
    pts[:, 1] += cy + 0.1
    pts += rng.normal(0, noise, pts.shape)
    return pts.astype(np.float32)


def place(hand: np.ndarray, style: SignerStyle) -> tuple[np.ndarray, np.ndarray]:
    """Put a right-handed hand into the (2, 21, 3) slot layout, mirroring for left-handed signers."""
    raw = np.zeros((2, 21, 3), np.float32)
    mask = np.zeros(2, bool)
    if style.left_handed:
        h = hand.copy()
        h[:, 0] = 1.0 - h[:, 0]
        raw[1], mask[1] = h, True
    else:
        raw[0], mask[0] = hand, True
    return raw, mask


def random_unknown_pose(rng: np.random.Generator) -> np.ndarray:
    """Intermediate, non-committal hand posture for the ``unknown`` rejection class."""
    return np.concatenate([rng.uniform(0.25, 0.75, 5), [rng.uniform(0.0, 0.3)]])


def static_sample(lang: str, label: str, style: SignerStyle, rng: np.random.Generator):
    """One synthetic static sample ``(raw, mask)``."""
    pose = random_unknown_pose(rng) if label == UNKNOWN else np.array(STATIC_POSES[lang][label])
    hand = hand_from_pose(pose, style, rng, rot_extra_deg=rng.normal(0, 4),
                          centre=(style.centre[0] + rng.normal(0, 0.03), style.centre[1] + rng.normal(0, 0.03)))
    return place(hand, style)


def _lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


def sequence_sample(label: str, style: SignerStyle, rng: np.random.Generator, T: int = 30):
    """One synthetic dynamic sample ``(raw (T,2,21,3), mask (T,2))``."""
    speed = rng.uniform(0.8, 1.25)
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.8, 1.2)
    open_p, fist_p = np.array(POSES_ASL["B"]), np.array(POSES_ASL["A"])
    V = np.array(POSES_ASL["V"])
    raws, masks = [], []
    for t in range(T):
        u = t / (T - 1)
        cx, cy = style.centre
        if label == "WAVE":
            pose, cx = open_p, cx + 0.10 * amp * np.sin(2 * np.pi * 2 * speed * u + phase)
        elif label == "CIRCLE":
            pose = open_p
            cx += 0.08 * amp * np.cos(2 * np.pi * speed * u + phase)
            cy += 0.08 * amp * np.sin(2 * np.pi * speed * u + phase)
        elif label == "NOD":
            pose, cy = fist_p, cy + 0.08 * amp * np.sin(2 * np.pi * 1.5 * speed * u + phase)
        elif label == "SWIPE":
            pose, cx = V, cx - 0.2 * amp + 0.4 * amp * min(1.0, u * speed)
        elif label == "PINCH":
            pose = _lerp(open_p, fist_p, 0.5 - 0.5 * np.cos(np.pi * min(1.0, u * speed)))
        else:
            raise KeyError(label)
        hand = hand_from_pose(pose, style, rng, centre=(cx, cy), pose_noise=0.02, noise=0.003)
        r, m = place(hand, style)
        raws.append(r)
        masks.append(m)
    return np.stack(raws), np.stack(masks)


def build_static_dataset(lang: str, n_signers: int, per_class: int, seed: int):
    """Return ``(raw, mask, labels, signer_ids)`` for one language (includes ``unknown``)."""
    rng = np.random.default_rng(seed + (0 if lang == "ASL" else 1000))
    signers = make_signers(n_signers, rng)
    raws, masks, ys, sids = [], [], [], []
    for s, style in enumerate(signers):
        for label in list(STATIC_POSES[lang]) + [UNKNOWN]:
            for _ in range(per_class * (2 if label == UNKNOWN else 1)):
                r, m = static_sample(lang, label, style, rng)
                raws.append(r); masks.append(m); ys.append(label); sids.append(f"synth_{lang}_{s}")
    return np.stack(raws), np.stack(masks), np.array(ys), np.array(sids)


def build_dynamic_dataset(lang: str, n_signers: int, per_class: int, seed: int, T: int = 30):
    """Return ``(raw (N,T,2,21,3), mask (N,T,2), labels, signer_ids)``."""
    rng = np.random.default_rng(seed + (7 if lang == "ASL" else 1007))
    signers = make_signers(n_signers, rng)
    raws, masks, ys, sids = [], [], [], []
    for s, style in enumerate(signers):
        for label in DYNAMIC_LABELS:
            for _ in range(per_class):
                r, m = sequence_sample(label, style, rng, T)
                raws.append(r); masks.append(m); ys.append(label); sids.append(f"synth_{lang}_{s}")
    return np.stack(raws), np.stack(masks), np.array(ys), np.array(sids)


def hold_static(lang: str, label: str, style: SignerStyle, rng: np.random.Generator, n: int, jitter: float = 0.004):
    """A held static sign: ONE pose, plus small independent per-frame landmark jitter (like a real held hand)."""
    pose = random_unknown_pose(rng) if label == UNKNOWN else np.array(STATIC_POSES[lang][label])
    base = hand_from_pose(pose, style, rng, noise=0.0)
    return [place(base + rng.normal(0, jitter, base.shape).astype(np.float32), style) for _ in range(n)]
