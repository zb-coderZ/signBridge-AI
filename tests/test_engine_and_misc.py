import numpy as np
import pytest

from core.compat import ensure_import_order

ensure_import_order()
from core.dataset import LandmarkSet, cross_signer_split  # noqa: E402
from core.engine import RecognitionEngine  # noqa: E402
from core.language_router import LanguageRouter  # noqa: E402
from core.models_io import load_language_models  # noqa: E402
from core.practice import make_sign_animation, similarity  # noqa: E402
from core.synthetic import (DYNAMIC_LABELS, hold_static, make_signers, sequence_sample)  # noqa: E402
from core.tts import TTSEngine  # noqa: E402
from core.stt import transcribe_wav  # noqa: E402

MODELS = {l: load_language_models(l) for l in ("PSL", "ASL")}
needs_models = pytest.mark.skipif(not (MODELS["ASL"].static and MODELS["ASL"].dynamic),
                                  reason="run scripts/build_demo.py first")


def run_static(eng, lang, label, style, rng, secs=2.0, fps=20, t0=0.0):
    t, commits = t0, []
    for r, m in hold_static(lang, label, style, rng, int(secs * fps)):
        res = eng.process_landmarks(r, m, t)
        t += 1 / fps
        if res.committed:
            commits.append(res.committed)
    return t, commits


@needs_models
def test_static_commit_once_then_no_repeat_and_language_auto_detect():
    rng = np.random.default_rng(1)
    st = make_signers(1, rng)[0]
    eng = RecognitionEngine(MODELS)
    t, commits = run_static(eng, "ASL", "L", st, rng, secs=3.0)
    assert commits == ["L"]                                  # held 3 s -> exactly one commit
    assert eng.last.language == "ASL"                        # Auto picked the right language
    assert eng.history[-1]["sign"] == "L" and eng.sentence.text == "L"


@needs_models
def test_manual_language_overrides_auto():
    rng = np.random.default_rng(2)
    st = make_signers(1, rng)[0]
    eng = RecognitionEngine(MODELS)
    eng.router.set_mode("PSL")
    run_static(eng, "ASL", "L", st, rng, secs=1.0)
    assert eng.last.language == "PSL"


@needs_models
@pytest.mark.parametrize("left", [False, True])
def test_dynamic_sign_committed_for_right_and_left_handed(left):
    rng = np.random.default_rng(3)
    st = make_signers(1, rng)[0]
    st.left_handed = left
    eng = RecognitionEngine(MODELS)
    eng.router.set_mode("ASL")
    seq, mk = sequence_sample("CIRCLE", st, rng)
    commits, t = [], 0.0
    for i in range(90):
        res = eng.process_landmarks(seq[i % 30], mk[i % 30], t)
        t += 1 / 30
        if res.committed:
            commits.append(res.committed)
    assert "CIRCLE" in commits


@needs_models
def test_no_hand_resets_and_never_commits():
    eng = RecognitionEngine(MODELS)
    for i in range(50):
        res = eng.process_landmarks(np.zeros((2, 21, 3), np.float32), np.zeros(2, bool), i / 20)
        assert res.committed is None and res.prediction is None


def test_router_auto_picks_higher_average_and_manual_overrides():
    r = LanguageRouter(("PSL", "ASL"), window_s=3.0)
    for i in range(30):
        r.update("PSL", 0.4, i * 0.1)
        r.update("ASL", 0.9, i * 0.1)
    assert r.active(3.0) == "ASL"
    r.set_mode("PSL")
    assert r.active(3.0) == "PSL"
    r.set_mode("Auto")
    assert r.active(100.0) in ("PSL", "ASL")                 # old data expired, still returns something valid


def test_router_window_forgets_old_scores():
    r = LanguageRouter(("PSL", "ASL"), window_s=3.0)
    for i in range(30):
        r.update("ASL", 0.9, i * 0.1)
    for i in range(30):
        r.update("PSL", 0.95, 10 + i * 0.1)
    assert r.active(13.0) == "PSL"


def test_practice_similarity_and_feedback():
    ref = np.random.default_rng(0).normal(size=(21, 3)).astype(np.float32)
    assert similarity(ref, ref).score == pytest.approx(100.0)
    bad = ref.copy()
    bad[8] = bad[5] + (bad[8] - bad[5]) * 0.1                 # shorten the index tip
    res = similarity(bad, ref)
    assert res.score < 100 and any("index" in f for f in res.feedback)


def test_sign_animation_handles_missing_letters():
    refs = {"static_labels": np.array(["A", "B"]), "static_refs": np.zeros((2, 21, 3), np.float32)}
    gif, notes = make_sign_animation("ABZ", refs)
    assert gif and gif[:3] == b"GIF" and any("'Z'" in n for n in notes)
    gif, notes = make_sign_animation("ZZ", refs)
    assert gif is None and notes


def test_cross_signer_split_has_no_signer_overlap_and_falls_back():
    y = np.array(["a", "b"] * 30)
    sig = np.repeat([f"s{i}" for i in range(10)], 6)
    tr, te, mode = cross_signer_split(y, sig, 0.3, 0)
    assert mode == "cross_signer" and not set(sig[tr]) & set(sig[te])
    _, _, mode2 = cross_signer_split(y, np.array(["unknown"] * 60), 0.3, 0)
    assert mode2 == "random_NOT_cross_signer"


def test_landmarkset_roundtrip(tmp_path):
    ds = LandmarkSet(np.zeros((3, 2, 21, 3), np.float32), np.ones((3, 2), bool), np.array(["a", "b", "a"]),
                     np.array(["x", "y", "z"]), "ASL", "static", {"k": 1})
    ds.save(tmp_path / "d.npz")
    back = LandmarkSet.load(tmp_path / "d.npz")
    assert len(back) == 3 and back.lang == "ASL" and back.meta == {"k": 1}


def test_tts_never_raises_and_reports_status():
    tts = TTSEngine()
    status = tts.speak("hello", "en")
    assert isinstance(status, str) and status
    assert isinstance(tts.speak("   ", "en"), str)
    assert isinstance(tts.speak("salam", "ur"), str)         # no Urdu voice must not crash


def test_stt_bad_audio_returns_error_not_exception():
    r = transcribe_wav(b"not a wav", "en")
    assert r.text == "" and r.error
