"""Headless Streamlit smoke test. The WebRTC component needs a real browser session, so only
``webrtc_streamer`` is stubbed; all other tabs run for real."""
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from core.compat import ensure_import_order  # noqa: E402

ensure_import_order()
import streamlit_webrtc  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr(streamlit_webrtc, "webrtc_streamer",
                        lambda **kw: SimpleNamespace(state=SimpleNamespace(playing=False)))
    return AppTest.from_file("../app/streamlit_app.py", default_timeout=180).run()


def test_app_starts_without_exceptions(app):
    assert not app.exception
    assert len(app.tabs) == 5


def test_app_explains_missing_image_model(app):
    assert not app.exception
    if app.error:
        assert any("No trained image model" in e.value for e in app.error)
    else:
        assert any("Classify one sign image" in h.value for h in app.subheader)


def test_app_exposes_required_tabs(app):
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Image", "Video", "Live Webcam", "Data Mining Insights", "History"]
