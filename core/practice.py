"""Practice mode (landmark similarity + feedback) and reference rendering for the two-way mode."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.config import load_config
from core.landmarks import FINGER_NAMES, FINGER_TIPS, HAND_CONNECTIONS


@dataclass
class PracticeResult:
    """Similarity of an attempt against a reference."""
    score: float                 # 0..100
    distance: float              # mean landmark distance, palm units
    feedback: list[str]


def similarity(user_norm: np.ndarray, ref_norm: np.ndarray, max_distance: Optional[float] = None) -> PracticeResult:
    """Compare two canonical normalised hands ``(21, 3)``.

    score = 100 * max(0, 1 - d / d_max), d = mean Euclidean landmark distance in palm units
    (a transparent heuristic, not a calibrated probability). Feedback compares per-finger
    tip-to-wrist reach, which captures "extend more / curl more".
    """
    dmax = max_distance or load_config()["practice"]["max_distance"]
    d = float(np.linalg.norm(user_norm - ref_norm, axis=-1).mean())
    score = 100.0 * max(0.0, 1.0 - d / dmax)
    tips_u = np.linalg.norm(user_norm[FINGER_TIPS], axis=-1)
    tips_r = np.linalg.norm(ref_norm[FINGER_TIPS], axis=-1)
    fb = []
    for name, u, r in zip(FINGER_NAMES, tips_u, tips_r):
        if u < r - 0.18:
            fb.append(f"Extend your {name} finger more.")
        elif u > r + 0.18:
            fb.append(f"Curl your {name} finger a little more.")
    if not fb:
        fb.append("Finger positions match well." if score >= 70 else "Fingers look right - adjust the hand angle or orientation.")
    prefix = "Excellent!" if score >= 85 else "Good - almost there." if score >= 65 else "Keep trying."
    return PracticeResult(score, d, [prefix] + fb)


# ------------------------------------------------------------------------------ rendering
def render_hand(norm_hand: np.ndarray, size: int = 320, title: str = "", offset: tuple[float, float] = (0.0, 0.0),
                color=(40, 160, 90)) -> np.ndarray:
    """Draw a canonical normalised hand ``(21, 3)`` as an RGB skeleton image."""
    img = np.full((size, size, 3), 250, np.uint8)
    pts = norm_hand[:, :2] * (size * 0.30)
    pts = pts + np.array([size / 2 + offset[0] * size * 0.30, size * 0.82 + offset[1] * size * 0.30])
    p = pts.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, 3, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, tuple(q), 5, (30, 30, 30), -1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
    return img


def make_sign_animation(word: str, refs: dict, size: int = 320, ms_per_frame: int = 450):
    """Build an animated GIF (bytes) showing ``word`` as a sign.

    * exact match with a dynamic label -> the stored prototype motion;
    * otherwise fingerspelling: one reference hand per letter that has a static reference.
    Returns ``(gif_bytes | None, notes)``.
    """
    notes: list[str] = []
    frames: list[np.ndarray] = []
    word_u = word.strip().upper()
    dyn = {str(l).upper(): i for i, l in enumerate(refs.get("dynamic_labels", []))}
    stat = {str(l).upper(): i for i, l in enumerate(refs.get("static_labels", []))}
    if word_u in dyn:
        i = dyn[word_u]
        for hand, traj in zip(refs["dynamic_protos"][i], refs["dynamic_trajs"][i]):
            frames.append(render_hand(hand, size, word_u, offset=(float(traj[0]), float(traj[1]))))
        ms_per_frame = 70
    else:
        for ch in word_u:
            if ch.isspace():
                continue
            if ch in stat:
                img = render_hand(refs["static_refs"][stat[ch]], size, ch)
                frames += [img, img]
            else:
                notes.append(f"No reference for '{ch}' in this language's model - skipped.")
    if not frames:
        return None, notes or ["Nothing to show."]
    buf = io.BytesIO()
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:], duration=ms_per_frame, loop=0)
    return buf.getvalue(), notes
