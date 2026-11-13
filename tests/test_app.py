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
    assert len(app.tabs) == 6


def test_demo_simulation_commits_a_sign(app):
    app.selectbox(key="d_lang").select("ASL").run()      # label options depend on the language
    app.selectbox(key="d_label").select("L")
    next(b for b in app.button if "Simulate" in b.label).click()
    app.run()
    assert not app.exception
    assert any("Committed: L" in s.value for s in app.success)


def test_practice_tab_scores_simulated_attempt(app):
    assert not app.exception
    # the simulated-attempt section always renders a reference image and feedback bullets
    assert any("Simulated attempt" in m.value for m in app.markdown)
