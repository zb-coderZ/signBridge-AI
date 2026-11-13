from core.stability import StabilityFilter


def feed(f, label, conf, n, t0=0.0, dt=0.05):
    out = []
    for i in range(n):
        out.append(f.update(label, conf, t0 + i * dt))
    return out


def test_commits_after_n_frames():
    f = StabilityFilter(n_frames=5, threshold=0.7, cooldown_s=1.0)
    out = feed(f, "A", 0.9, 5)
    assert out[:4] == [None] * 4 and out[4] == "A"


def test_low_confidence_never_commits():
    f = StabilityFilter(n_frames=3, threshold=0.7)
    assert all(o is None for o in feed(f, "A", 0.5, 30))


def test_unknown_is_rejected():
    f = StabilityFilter(n_frames=3, threshold=0.7)
    assert all(o is None for o in feed(f, "unknown", 0.99, 30))


def test_interrupted_streak_restarts():
    f = StabilityFilter(n_frames=4, threshold=0.7)
    assert feed(f, "A", 0.9, 3) == [None] * 3
    f.update("A", 0.2, 0.2)
    assert feed(f, "A", 0.9, 3, t0=0.3) == [None] * 3


def test_changing_label_restarts():
    f = StabilityFilter(n_frames=3, threshold=0.7)
    feed(f, "A", 0.9, 2)
    assert feed(f, "B", 0.9, 3, t0=0.5)[-1] == "B"


def test_holding_does_not_repeat_even_after_cooldown():
    f = StabilityFilter(n_frames=3, threshold=0.7, cooldown_s=0.5)
    out = feed(f, "A", 0.9, 100, dt=0.05)         # 5 s of holding "A"
    assert out.count("A") == 1


def test_release_then_same_sign_commits_again_after_cooldown():
    f = StabilityFilter(n_frames=3, threshold=0.7, cooldown_s=0.5)
    assert feed(f, "A", 0.9, 3)[-1] == "A"
    f.reset()                                      # hand removed
    assert feed(f, "A", 0.9, 3, t0=2.0)[-1] == "A"


def test_cooldown_blocks_fast_different_sign():
    f = StabilityFilter(n_frames=3, threshold=0.7, cooldown_s=2.0)
    assert feed(f, "A", 0.9, 3, t0=0.0)[-1] == "A"
    assert all(o is None for o in feed(f, "B", 0.9, 5, t0=0.2))
    assert "B" in feed(f, "B", 0.9, 3, t0=2.5)   # streak already built; commits once cooldown ends


def test_without_release_requirement_repeats_after_cooldown():
    f = StabilityFilter(n_frames=2, threshold=0.7, cooldown_s=0.5, require_release=False)
    out = feed(f, "A", 0.9, 60, dt=0.05)
    assert out.count("A") >= 2


def test_progress():
    f = StabilityFilter(n_frames=4, threshold=0.7)
    feed(f, "A", 0.9, 2)
    assert f.progress == ("A", 0.5)
