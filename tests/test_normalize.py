import numpy as np
import pytest

from core.normalize import (DYNAMIC_DIM, STATIC_DIM, augment_raw, mirror_raw, motion_score,
                            normalize_hands, sequence_features, static_features)


def make_hand(seed=0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.3, 0.7, (21, 3)).astype(np.float32)
    h[0] = [0.5, 0.8, 0.0]
    h[9] = [0.5, 0.55, 0.0]
    return h


def pack(h, slot=0):
    raw = np.zeros((2, 21, 3), np.float32)
    mask = np.zeros(2, bool)
    raw[slot], mask[slot] = h, True
    return raw, mask


def test_translation_invariance():
    raw, mask = pack(make_hand())
    shifted = raw.copy()
    shifted[0] += np.array([0.1, -0.2, 0.0], np.float32)
    assert np.allclose(normalize_hands(raw, mask), normalize_hands(shifted, mask), atol=1e-5)


def test_scale_invariance():
    raw, mask = pack(make_hand())
    scaled = raw.copy()
    scaled[0] = scaled[0, 0] + (scaled[0] - scaled[0, 0]) * 1.7
    assert np.allclose(normalize_hands(raw, mask), normalize_hands(scaled, mask), atol=1e-4)


def test_wrist_is_origin_and_palm_is_unit():
    raw, mask = pack(make_hand())
    n = normalize_hands(raw, mask)[0]
    assert np.allclose(n[0], 0)
    assert np.isclose(np.linalg.norm(n[9]), 1.0, atol=1e-5)


def test_left_hand_mirrored_to_canonical_right():
    h = make_hand()
    right, mr = pack(h, 0)
    left_h = h.copy()
    left_h[:, 0] = 1.0 - left_h[:, 0]            # a true mirror image, in the Left slot
    left, ml = pack(left_h, 1)
    nr = normalize_hands(right, mr)[0]
    nl = normalize_hands(left, ml)[0]            # lone Left hand is moved to slot 0 after mirroring
    assert np.allclose(nr, nl, atol=1e-5)


def test_absent_hand_is_zero_and_feature_dims():
    raw, mask = pack(make_hand())
    assert np.all(normalize_hands(raw, mask)[1] == 0)
    f = static_features(raw, mask)
    assert f.shape == (STATIC_DIM,) and f[-2:].tolist() == [1.0, 0.0]


def test_degenerate_hand_does_not_nan():
    raw = np.zeros((2, 21, 3), np.float32)
    f = static_features(raw, np.array([True, True]))
    assert np.isfinite(f).all()


def test_sequence_features_shape_and_translation_invariance():
    T = 30
    base = np.stack([pack(make_hand())[0]] * T)
    mask = np.tile(np.array([True, False]), (T, 1))
    base[:, 0, :, 0] += np.linspace(0, 0.1, T)[:, None]
    f1 = sequence_features(base, mask)
    f2 = sequence_features(base + np.array([0.05, 0.05, 0], np.float32), mask)
    assert f1.shape == (T, DYNAMIC_DIM)
    assert np.allclose(f1, f2, atol=1e-4)
    assert motion_score(base, mask) > 0.1          # travelling hand


def test_motion_score_zero_for_static():
    T = 10
    raw = np.stack([pack(make_hand())[0]] * T)
    mask = np.tile(np.array([True, False]), (T, 1))
    assert motion_score(raw, mask) == pytest.approx(0.0, abs=1e-6)


def test_mirror_is_involution():
    raw, _ = pack(make_hand())
    assert np.allclose(mirror_raw(mirror_raw(raw)), raw, atol=1e-6)


def test_augmentation_shapes_and_determinism():
    raws = np.stack([pack(make_hand(i))[0] for i in range(5)])
    masks = np.tile(np.array([True, False]), (5, 1))
    a1, m1 = augment_raw(raws, masks, np.random.default_rng(1), copies=3)
    a2, _ = augment_raw(raws, masks, np.random.default_rng(1), copies=3)
    assert a1.shape == (15, 2, 21, 3) and m1.shape == (15, 2)
    assert np.allclose(a1, a2)
    seqs = np.stack([raws] * 4, axis=1)                       # (5, 4, 2, 21, 3)
    s, sm = augment_raw(seqs, np.tile(masks[:, None], (1, 4, 1)), np.random.default_rng(0), copies=2)
    assert s.shape == (10, 4, 2, 21, 3) and sm.shape == (10, 4, 2)


def test_left_handed_sequence_equals_mirrored_right_handed_sequence():
    """Regression: shape AND trajectory must be mirrored for the Left slot."""
    T = 30
    rng = np.random.default_rng(3)
    hand = make_hand()
    seq_r = np.zeros((T, 2, 21, 3), np.float32)
    for t in range(T):
        h = hand.copy()
        h[:, 0] += 0.1 * np.cos(2 * np.pi * t / T)
        h[:, 1] += 0.1 * np.sin(2 * np.pi * t / T)
        seq_r[t, 0] = h
    mask_r = np.tile(np.array([True, False]), (T, 1))
    seq_l = np.zeros_like(seq_r)
    seq_l[:, 1] = seq_r[:, 0]
    seq_l[:, 1, :, 0] = 1.0 - seq_l[:, 1, :, 0]        # a real mirror image performed by the left hand
    mask_l = np.tile(np.array([False, True]), (T, 1))
    fr, fl = sequence_features(seq_r, mask_r), sequence_features(seq_l, mask_l)
    assert np.allclose(fr, fl, atol=1e-4)      # identical canonical features: shape, flags and trajectory


def test_single_hand_features_are_handedness_invariant():
    h = make_hand(2)
    right, mr = pack(h, 0)
    lh = h.copy()
    lh[:, 0] = 1.0 - lh[:, 0]
    left, ml = pack(lh, 1)
    assert np.allclose(static_features(right, mr), static_features(left, ml), atol=1e-5)
