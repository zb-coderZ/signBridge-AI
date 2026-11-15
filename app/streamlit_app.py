"""SignBridge AI image, video, and live webcam application."""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compat import ensure_import_order  # noqa: E402

ensure_import_order(need_tf=True, webrtc=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

from app.components.image_processor import ImageVideoProcessor  # noqa: E402
from app.ui_helpers import csv_bytes, hero, top3_bars  # noqa: E402
from core.config import load_config, path_of, setup_logging  # noqa: E402
from core.image_dataset import find_dataset  # noqa: E402
from core.image_engine import ImageRecognitionEngine, ImageResult  # noqa: E402
from core.image_models import ImageModel  # noqa: E402
from core.tts import TTSEngine  # noqa: E402

st.set_page_config(page_title="SignBridge AI", page_icon="🤟", layout="wide")
setup_logging(to_file=True)
CFG = load_config()


@st.cache_resource(show_spinner="Loading trained CNN and baseline...")
def get_model() -> ImageModel | None:
    try:
        return ImageModel(path_of("image_models"))
    except Exception as exc:
        st.session_state["model_error"] = str(exc)
        return None


@st.cache_resource
def get_tts() -> TTSEngine:
    return TTSEngine()


def get_engine(model: ImageModel | None) -> ImageRecognitionEngine | None:
    if model is None or not model.available:
        return None
    if "image_engine" not in st.session_state:
        try:
            engine = ImageRecognitionEngine(model, CFG["recognition"]["confidence_threshold"],
                                            CFG["recognition"]["stable_frames"], CFG["recognition"]["cooldown_s"])
            engine.on_commit.append(lambda label, confidence: get_tts().speak(label, "en")
                                    if engine.speak_commits else None)
            st.session_state.image_engine = engine
        except Exception as exc:
            st.session_state["engine_error"] = str(exc)
            return None
    return st.session_state.image_engine


def show_result(result: ImageResult, key: str, allow_add: bool = True) -> None:
    if result.error:
        st.warning(result.error)
        return
    if result.prepared is not None:
        st.image(result.prepared.preview, caption="28x28 input seen by the model", width=220, clamp=True)
    if result.top3:
        st.markdown(f"### Prediction: `{result.prediction}` ({result.confidence * 100:.1f}%)")
        top3_bars(result.top3)
        st.caption(f"Preprocessing and inference latency: {result.latency_ms:.1f} ms")
        if allow_add and st.button(f"Add {result.prediction} to sentence", key=f"add_{key}"):
            engine = st.session_state.image_engine
            engine.sentence.add(result.prediction)
            st.toast(f"Added {result.prediction}")


def sentence_panel(engine: ImageRecognitionEngine, tts: TTSEngine, key: str) -> None:
    st.markdown(f"**Running sentence:** `{engine.sentence.text or '(empty)'}`")
    c1, c2, c3 = st.columns(3)
    if c1.button("Space", key=f"space_{key}"):
        engine.sentence.space()
    if c2.button("Backspace", key=f"back_{key}"):
        engine.sentence.backspace()
    if c3.button("Clear", key=f"clear_{key}"):
        engine.sentence.clear()
    if st.button("Speak sentence", key=f"speak_{key}"):
        st.info(tts.speak(engine.sentence.text, "en"))


def classify_bytes(engine: ImageRecognitionEngine, data: bytes, key: str) -> ImageResult:
    array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        result = ImageResult(error="The uploaded file is not a readable image.")
    else:
        result = engine.process(array, model_name="cnn")
    show_result(result, key)
    return result


model = get_model()
engine = get_engine(model)
tts = get_tts()
st.session_state["speak_commits"] = st.sidebar.checkbox("Speak committed letters", False)
if engine is not None:
    engine.speak_commits = st.session_state["speak_commits"]
live_model = st.sidebar.selectbox("Live model", ["random_forest", "cnn"], index=0,
                                  help="Random Forest is faster and scored higher on the official test split.")
hero()

if model is None or not model.available:
    st.error("No trained image model is available. Train it with `python scripts/train_image.py`.")
    if st.session_state.get("model_error"):
        st.caption(st.session_state["model_error"])
elif engine is None:
    st.error("The image engine could not start.")
    st.caption(st.session_state.get("engine_error", "Check the MediaPipe hand model and logs."))

tab_image, tab_video, tab_live, tab_mining, tab_history = st.tabs(
    ["Image", "Video", "Live Webcam", "Data Mining Insights", "History"])

with tab_image:
    st.subheader("Classify one sign image")
    st.caption("Upload a photo with one clearly visible hand. MediaPipe crops it before CNN inference.")
    uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg", "bmp"], key="image_upload")
    if uploaded is not None and engine is not None:
        classify_bytes(engine, uploaded.getvalue(), "image")
        sentence_panel(engine, tts, "image_sentence")

with tab_video:
    st.subheader("Classify a sign-language video")
    video = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"], key="video_upload")
    stride = st.slider("Process every Nth frame", 1, 10, 3)
    if video is not None and engine is not None and st.button("Process video"):
        engine.reset()
        with tempfile.NamedTemporaryFile(suffix=Path(video.name).suffix, delete=False) as handle:
            handle.write(video.getvalue())
            video_path = handle.name
        cap = cv2.VideoCapture(video_path)
        frame_i, processed, last = 0, 0, None
        progress = st.progress(0.0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if frame_i % stride:
                continue
            last = engine.process(frame, timestamp_ms=frame_i * 33)
            processed += 1
            if processed % 10 == 0:
                progress.progress(min(0.99, processed / max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))))
        cap.release()
        progress.progress(1.0)
        if last:
            show_result(last, "video", allow_add=False)
        st.success(f"Processed {processed} frames. Committed letters: {engine.sentence.text or '(none)'}")
        sentence_panel(engine, tts, "video_sentence")

with tab_live:
    st.subheader("Live webcam")
    if engine is None:
        st.info("Train the CNN first, then restart the app.")
    else:
        ctx = webrtc_streamer(
            key="signbridge-image", mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: ImageVideoProcessor(engine, live_model),
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": CFG["webrtc"]["stun_servers"]}]},
            async_processing=True)
        if not ctx.state.playing:
            st.info("Click START and allow browser camera permission.")
        @st.fragment(run_every=0.7)
        def live_result() -> None:
            result = engine.last
            show_result(result, "live", allow_add=False)
            sentence_panel(engine, tts, "live_sentence")
        live_result()

with tab_mining:
    st.subheader("Data Mining Insights")
    summary_path = path_of("reports") / "mining_summary.json"
    if not summary_path.exists():
        st.info("Run `python scripts/data_mining.py` after training to generate the real-data insights.")
    else:
        summary = pd.read_json(summary_path, typ="series")
        st.json(summary.to_dict())
        mining_dir = path_of("reports") / "mining"
        for name, caption in [("class_balance.png", "Class balance"), ("pca.png", "PCA"), ("tsne.png", "t-SNE"),
                              ("rf_feature_importance.png", "Random Forest pixel importance"),
                              ("cnn_saliency.png", "CNN saliency"), ("outliers.png", "Outliers")]:
            image = mining_dir / name
            if image.exists():
                st.image(str(image), caption=caption, width="stretch")
        st.caption("CNN saliency is represented by the trained CNN and processed-image preview; all reported statistics use real CSV rows.")

with tab_history:
    st.subheader("History")
    rows = engine.history if engine is not None else []
    frame = pd.DataFrame(rows, columns=["timestamp", "sign", "confidence", "source"])
    st.dataframe(frame, width="stretch")
    st.download_button("Export history CSV", csv_bytes(frame), "signbridge_history.csv", "text/csv", disabled=frame.empty)
    if engine is not None:
        sentence_panel(engine, tts, "history_sentence")
