"""Landmark data structures and helpers (no MediaPipe import needed here).

Canonical raw layout used everywhere in the project:

* ``raw``  : float32 array ``(..., 2, 21, 3)``  - slot 0 = Right hand, slot 1 = Left hand,
  21 MediaPipe landmarks, (x, y, z) in normalised image coordinates.
* ``mask`` : bool array ``(..., 2)``            - True where that hand is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

NUM_LANDMARKS = 21
SLOT_RIGHT, SLOT_LEFT = 0, 1

LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_TIPS = [4, 8, 12, 16, 20]
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandObservation:
    """One detected hand."""
    landmarks: np.ndarray      # (21, 3) normalised image coords
    handedness: str            # "Left" | "Right"
    score: float = 1.0


def observations_to_slots(hands: Sequence[HandObservation]) -> tuple[np.ndarray, np.ndarray]:
    """Place detected hands into the fixed (Right, Left) slots.

    If both detections claim the same handedness, the second one takes the free slot.
    """
    raw = np.zeros((2, NUM_LANDMARKS, 3), np.float32)
    mask = np.zeros(2, bool)
    for h in sorted(hands, key=lambda o: -o.score)[:2]:
        slot = SLOT_LEFT if h.handedness.lower().startswith("l") else SLOT_RIGHT
        if mask[slot]:
            slot = 1 - slot
        if mask[slot]:
            continue
        raw[slot] = np.asarray(h.landmarks, np.float32)[:NUM_LANDMARKS]
        mask[slot] = True
    return raw, mask


def resample_sequence(frames: list[tuple[np.ndarray, np.ndarray]], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly sample ``length`` frames from a list of ``(raw, mask)`` pairs.

    Shorter clips are padded by repeating nearest frames (linspace indices).
    """
    if not frames:
        return np.zeros((length, 2, NUM_LANDMARKS, 3), np.float32), np.zeros((length, 2), bool)
    idx = np.linspace(0, len(frames) - 1, length).round().astype(int)
    raw = np.stack([frames[i][0] for i in idx]).astype(np.float32)
    mask = np.stack([frames[i][1] for i in idx]).astype(bool)
    return raw, mask
