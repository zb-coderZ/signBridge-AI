"""SignBridge AI - Streamlit application.

Run:  streamlit run app/streamlit_app.py     (or run_app.bat / run_app.sh)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compat import ensure_import_order  # noqa: E402

ensure_import_order()                          # streamlit_webrtc -> mediapipe -> tensorflow (order matters!)

import platform  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

from app.components.video_processor import SignProcessor  # noqa: E402
from app.ui_helpers import conf_chart, csv_bytes, demo_banner, hero, history_table, top3_bars  # noqa: E402
from core.config import load_config, path_of, setup_logging  # noqa: E402
from core.dataset_info import DATASETS  # noqa: E402
from core.engine import RecognitionEngine, Settings  # noqa: E402
from core.hand_detector import DetectorUnavailable, HandDetector, draw_skeleton_arrays, ensure_hand_model  # noqa: E402
from core.landmarks import FINGER_NAMES, observations_to_slots  # noqa: E402
from core.models_io import load_language_models  # noqa: E402
from core.normalize import canonical_mask, normalize_hands, static_features  # noqa: E402
from core.practice import make_sign_animation, render_hand, similarity  # noqa: E402
from core.sentence_builder import gloss_to_sentence  # noqa: E402
from core.stt import transcribe_wav, whisper_available  # noqa: E402
from core.synthetic import hold_static, make_signers, sequence_sample  # noqa: E402
from core.tts import TTSEngine  # noqa: E402

st.set_page_config(page_title="SignBridge AI", page_icon="🤟", layout="wide")
setup_logging(to_file=True)
CFG = load_config()


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return {lang: load_language_models(lang) for lang in CFG["languages"]}


@st.cache_resource
def get_tts() -> TTSEngine:
    return TTSEngine()


@st.cache_resource(ttl=60, show_spinner=False)
def get_snapshot_detector():
    try:
        return HandDetector("image"), ""
    except Exception as exc:
        return None, str(exc)


def make_engine(models) -> RecognitionEngine:
    eng = RecognitionEngine(models)
    eng.speak, eng.tts_lang = False, "en"

    def _speak(label: str, lang: str, conf: float) -> None:
        if eng.speak:
            get_tts().speak(label, eng.tts_lang)

    eng.on_commit.append(_speak)
    return eng


models = get_models()
if "engine" not in st.session_state:
    st.session_state.engine = make_engine(models)
    st.session_state.demo_engine = make_engine(models)
    st.session_state.replies = []
engine: RecognitionEngine = st.session_state.engine
demo_engine: RecognitionEngine = st.session_state.demo_engine
tts = get_tts()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    lang_mode = st.selectbox("Language", ["Auto", "PSL", "ASL"], help="Auto picks the language with the higher average "
                             "confidence over the last 3 s. Choosing PSL/ASL overrides Auto.")
    sign_mode = st.selectbox("Sign type", ["Auto", "Static only", "Dynamic only"], help="Auto switches by detected hand motion.")
    thr = st.slider("Confidence threshold (static)", 0.3, 0.99, CFG["recognition"]["confidence_threshold"], 0.01)
    dthr = st.slider("Confidence threshold (dynamic)", 0.3, 0.99, CFG["recognition"]["dynamic_confidence_threshold"], 0.01)
    nfr = st.slider("Stable frames before commit (N)", 2, 30, CFG["recognition"]["stable_frames"])
    cool = st.slider("Cooldown (s)", 0.0, 4.0, CFG["recognition"]["cooldown_s"], 0.1)
    mthr = st.slider("Motion threshold (static/dynamic switch)", 0.02, 0.6, CFG["recognition"]["motion_threshold"], 0.01)
    skel = st.checkbox("Show landmark skeleton", True)
    st.divider()
    speak = st.checkbox("Speak each committed sign", False)
    voice_lang = st.radio("Speech language", ["English", "Urdu"], horizontal=True)
    use_llm = st.checkbox("Use LLM for sentence clean-up", False,
                          help=f"Optional. Needs the {CFG['llm']['env_key']} environment variable; falls back to rules.")
    st.caption("TTS: " + ("ready" if tts.available in (None, True) else "unavailable on this machine"))
    settings = Settings(lang_mode, sign_mode, thr, dthr, nfr, cool, mthr, skel)
    for e in (engine, demo_engine):
        e.apply_settings(settings)
    engine.speak = speak
    engine.tts_lang = "ur" if voice_lang == "Urdu" else "en"
    demo_engine.speak = False

hero()
demo_banner(models)
if not engine.has_models:
    st.error("No recognition models found. Run `python scripts/build_demo.py` (demo model) or train on real data - see the Status tab.")

tab_live, tab_demo, tab_two, tab_prac, tab_hist, tab_status = st.tabs(
    ["🎥 Live translate", "🧪 Demo (no camera)", "↔️ Two-way", "🎯 Practice", "📋 History", "⚙️ Status"])


# ------------------------------------------------------------------ shared panels
def sentence_controls(eng: RecognitionEngine, key: str) -> None:
    snap = eng.snapshot()
    st.markdown("**Sentence**")
    st.markdown(f'<div class="sb-sentence">{snap["text"] or "&nbsp;"}</div>', unsafe_allow_html=True)
    c = st.columns(5)
    if c[0].button("␣ Space", key=f"sp{key}"):
        eng.sentence.space()
    if c[1].button("⌫ Back", key=f"bk{key}"):
        eng.sentence.backspace()
    if c[2].button("Clear", key=f"cl{key}"):
        eng.sentence.clear()
    if c[3].button("🔊 Speak", key=f"sk{key}"):
        st.toast(tts.speak(eng.sentence.text, eng.tts_lang))
    if c[4].button("✨ Clean up", key=f"cu{key}"):
        out, how = gloss_to_sentence(eng.sentence.text, CFG, use_llm)
        st.session_state[f"clean{key}"] = f"{out}  _(via {how})_"
    if st.session_state.get(f"clean{key}"):
        st.success(st.session_state[f"clean{key}"])


def prediction_panel(snap: dict) -> None:
    res = snap["last"]
    p = res.prediction
    if p is None:
        st.markdown('<div class="sb-pred">—</div>', unsafe_allow_html=True)
        st.caption(res.note or ("Show a hand to the camera." if res.hands == 0 else "Waiting for enough frames..."))
        return
    st.markdown(f'<div class="sb-pred">{p.label}</div>', unsafe_allow_html=True)
    st.caption(f"{p.language} · {p.kind} · confidence {p.confidence * 100:.1f}% · motion {res.motion:.2f}")
    st.progress(res.progress, text="stability (commits at 100%)")
    top3_bars(p.top)
    if len(p.per_language_conf) > 1:
        st.caption("Auto-detect confidences: " + ", ".join(f"{l} {c * 100:.0f}%" for l, c in p.per_language_conf.items()))


# ------------------------------------------------------------------ LIVE TAB
with tab_live:
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(
            key="signbridge", mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: SignProcessor(engine),
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": CFG["webrtc"]["stun_servers"]}]},
            async_processing=True)
        if not ctx.state.playing:
            st.info("Click **START**, allow camera access, and sign in front of the camera. No webcam? Use the **Demo (no camera)** "
                    "tab or the snapshot option below.")
        with st.expander("No live video? Classify a webcam snapshot instead"):
            shot = st.camera_input("Take a picture of a hand sign")
            if shot is not None:
                det, err = get_snapshot_detector()
                if det is None:
                    st.error("Hand model unavailable:\n\n" + err)
                else:
                    img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    if CFG["hand_detector"]["mirror_input"]:
                        img = cv2.flip(img, 1)
                    raw, mask = observations_to_slots(det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    if not mask.any():
                        st.warning("No hand found in the snapshot.")
                    else:
                        for lang, lm in models.items():
                            if lm.static:
                                probs = lm.static.predict(static_features(raw, mask))
                                top = sorted(zip(lm.static.classes, probs), key=lambda x: -x[1])[:3]
                                st.write(f"**{lang}**: " + ", ".join(f"{l} {p * 100:.0f}%" for l, p in top))
    with right:
        @st.fragment(run_every=0.5)
        def live_panel() -> None:
            snap = engine.snapshot()
            if snap["detector_error"]:
                st.error("Hand detector problem:\n\n" + snap["detector_error"])
            prediction_panel(snap)
            c1, c2 = st.columns(2)
            c1.metric("FPS", f"{snap['fps']:.0f}")
            c2.metric("Frame latency", f"{snap['latency_ms']:.1f} ms")
            conf_chart(snap["conf"])
            sentence_controls(engine, "live")
        live_panel()

# ------------------------------------------------------------------ DEMO TAB
with tab_demo:
    st.markdown("Feeds **synthetic hand landmarks** through the real pipeline (normalisation → model → language router → "
                "stability filter → sentence builder). No camera or MediaPipe model is needed.")
    c1, c2, c3, c4 = st.columns(4)
    d_lang = c1.selectbox("Language", CFG["languages"], key="d_lang")
    d_kind = c2.selectbox("Sign type", ["static", "dynamic"], key="d_kind")
    lm = models[d_lang]
    comp = getattr(lm, d_kind)
    labels = [c for c in (comp.classes if comp else []) if c != CFG["recognition"]["unknown_label"]]
    d_label = c3.selectbox("Sign to simulate", labels or ["(no model)"], key="d_label")
    d_left = c4.checkbox("Left-handed signer", False)
    seed = st.number_input("Virtual signer seed", 0, 9999, 7)
    if st.button("▶ Simulate signing", disabled=comp is None):
        rng = np.random.default_rng(int(seed))
        style = make_signers(1, rng)[0]
        style.left_handed = d_left
        demo_engine.settings.language_mode = d_lang
        demo_engine.router.set_mode(d_lang)
        demo_engine.settings.sign_mode = "Auto"
        frames, t, rows = [], 0.0, []
        if d_kind == "static":
            frames += hold_static(d_lang, d_label, style, rng, 45)
        else:
            seq, mk = sequence_sample(d_label, style, rng)
            frames += [(seq[i % 30], mk[i % 30]) for i in range(90)]
        frames += [(np.zeros((2, 21, 3), np.float32), np.zeros(2, bool))] * 4          # hand leaves the view
        last = None
        for raw, mask in frames:
            res = demo_engine.process_landmarks(raw, mask, t)
            t += 1 / 30
            if mask.any():
                last = (raw, mask)
            if res.prediction:
                rows.append({"t (s)": round(t, 2), "type": res.kind, "prediction": res.prediction.label,
                             "confidence": round(res.prediction.confidence, 3), "committed": res.committed or ""})
        if last is not None:
            canvas = draw_skeleton_arrays(np.full((300, 400, 3), 245, np.uint8), *last)
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Last simulated frame (synthetic landmarks)")
        commits = [r for r in rows if r["committed"]]
        st.success("Committed: " + (", ".join(r["committed"] for r in commits) or "nothing (confidence/stability not reached)"))
        with st.expander("Per-frame log"):
            st.dataframe(rows, width="stretch", height=240)
    sentence_controls(demo_engine, "demo")

# ------------------------------------------------------------------ TWO-WAY TAB
with tab_two:
    a, b = st.columns(2)
    with a:
        st.subheader("Hearing person → text")
        st.caption("Record a spoken reply (browser microphone). Transcription uses Google Web Speech (needs internet) or "
                   + ("offline Whisper." if whisper_available() else "offline Whisper if you `pip install faster-whisper`."))
        stt_lang = st.radio("Spoken language", ["English", "Urdu"], horizontal=True, key="stt_lang")
        audio = st.audio_input("Record reply")
        if audio is not None and st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                r = transcribe_wav(audio.getvalue(), "ur" if stt_lang == "Urdu" else "en")
            if r.text:
                st.session_state.replies.append({"time": time.strftime("%H:%M:%S"), "text": r.text, "via": r.backend})
            else:
                st.warning(r.error or "Nothing recognised.")
        typed = st.text_input("...or type the reply", key="typed_reply")
        if st.button("Show reply") and typed.strip():
            st.session_state.replies.append({"time": time.strftime("%H:%M:%S"), "text": typed.strip(), "via": "typed"})
        for rep in reversed(st.session_state.replies[-5:]):
            st.markdown(f'<div class="sb-sentence">{rep["text"]}</div>', unsafe_allow_html=True)
            st.caption(f'{rep["time"]} · {rep["via"]}')
    with b:
        st.subheader("Text → sign")
        t_lang = st.selectbox("Sign language", CFG["languages"], key="t_lang")
        word = st.text_input("Type a word or letters", "WAVE" if models[t_lang].dynamic else "")
        if st.button("Show sign") and word.strip():
            gif, notes = make_sign_animation(word, models[t_lang].references)
            if gif:
                st.image(gif, caption=f"'{word}' as reference landmarks (animated)")
            for n in notes:
                st.caption(n)
            if models[t_lang].is_demo:
                st.warning("Reference hands come from the DEMO model (synthetic shapes), not real signs.")

# ------------------------------------------------------------------ PRACTICE TAB
with tab_prac:
    p1, p2 = st.columns(2)
    pr_lang = p1.selectbox("Language", CFG["languages"], key="pr_lang")
    refs = models[pr_lang].references
    plabels = [str(x) for x in refs.get("static_labels", [])]
    if not plabels:
        st.info("No static reference signs for this language yet (train a static model).")
    else:
        pr_label = p2.selectbox("Target sign", plabels, key="pr_label")
        ref = refs["static_refs"][plabels.index(pr_label)]
        engine.practice_target = (pr_lang, pr_label) if st.toggle("Score my live camera hand (start the Live tab video)", False) else None
        c1, c2 = st.columns(2)
        c1.image(render_hand(ref, 300, f"Target {pr_label}"), caption="Reference")

        @st.fragment(run_every=0.7)
        def practice_live() -> None:
            res = engine.snapshot()["last"].practice
            if engine.practice_target and res:
                st.metric("Similarity", f"{res.score:.0f} / 100")
                st.progress(min(res.score / 100, 1.0))
                for line in res.feedback:
                    st.write("• " + line)
            elif engine.practice_target:
                st.caption("Waiting for a hand...")
        with c2:
            practice_live()
        st.divider()
        st.markdown("**Simulated attempt** (try the scoring without a camera)")
        s1, s2 = st.columns(2)
        finger = s1.selectbox("Finger that is wrong", FINGER_NAMES, index=1)
        amount = s2.slider("How much it is curled/shortened", 0.0, 0.9, 0.5, 0.05)
        base = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}[finger]
        att = ref.copy()
        idx = list(range(base + 1, base + 4)) if finger != "thumb" else [2, 3, 4]
        for i in idx:
            att[i] = att[base] + (att[i] - att[base]) * (1 - amount)
        sim = similarity(att, ref)
        d1, d2 = st.columns(2)
        d1.image(render_hand(att, 300, "Attempt", color=(200, 90, 40)), caption=f"Similarity {sim.score:.0f}/100")
        with d2:
            for line in sim.feedback:
                st.write("• " + line)

# ------------------------------------------------------------------ HISTORY TAB
with tab_hist:
    @st.fragment(run_every=1.0)
    def history_panel() -> None:
        df = history_table(engine.snapshot()["history"] + demo_engine.snapshot()["history"])
        df = df.sort_values("timestamp") if len(df) else df
        st.dataframe(df, width="stretch", height=360)
        st.download_button("⬇ Export CSV", csv_bytes(df), "signbridge_history.csv", "text/csv", disabled=df.empty)
    history_panel()
    if st.button("Clear history"):
        engine.history.clear(); demo_engine.history.clear()

# ------------------------------------------------------------------ STATUS TAB
with tab_status:
    st.subheader("Environment")
    st.write(f"Python {platform.python_version()} · {platform.system()}")
    hm = path_of("hand_model")
    if hm.exists():
        st.success(f"Hand landmark model present: {hm.name}")
    else:
        st.warning(f"Hand landmark model missing ({hm}). It is downloaded automatically on first camera use (needs internet).")
        if st.button("Try downloading now"):
            try:
                ensure_hand_model()
                st.success("Downloaded.")
            except DetectorUnavailable as exc:
                st.error(str(exc))
    st.write("TTS:", "available" if tts.available else f"unavailable ({tts.error or 'initialising'})",
             "· Urdu voice installed:" , "yes" if tts.has_urdu_voice() else "no (default voice is used)")
    st.subheader("Models")
    for lang, lm in models.items():
        for kind in ("static", "dynamic"):
            comp = getattr(lm, kind)
            if comp:
                st.write(f"**{lang} {kind}** — {'DEMO (synthetic)' if comp.is_demo else 'trained'} · default `{comp.default}` · "
                         f"{len(comp.classes)} classes · val acc {comp.meta.get('val_accuracy')} · split {comp.meta.get('split_mode')}")
        for prob in lm.problems:
            st.warning(prob)
    st.subheader("Datasets (data/raw)")
    for name, d in DATASETS.items():
        p = path_of("data_raw") / d["folder"]
        st.write(("✅ " if p.exists() and any(p.iterdir()) else "❌ ") + f"{name} → {p}")
    st.subheader("Evaluation results")
    for folder, title in ((path_of("reports"), "Real data"), (path_of("reports") / "demo", "Synthetic demo data")):
        f = folder / "results_table.md"
        if f.exists():
            st.markdown(f"**{title}**")
            st.markdown(f.read_text(encoding="utf-8"))
