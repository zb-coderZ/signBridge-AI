"""Import-order guard.

Empirically (Python 3.12, Linux) the process segfaults if ``mediapipe`` or ``tensorflow``
is imported BEFORE ``streamlit_webrtc``.  The project-wide order is therefore:

    streamlit_webrtc (if installed)  ->  mediapipe  ->  tensorflow

Call :func:`ensure_import_order` at the very top of any entry point.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")


def ensure_import_order(need_tf: bool = True, webrtc: bool = True) -> dict[str, bool]:
    """Import the heavy native libraries in the safe order; return which are available."""
    logging.getLogger("streamlit").setLevel(logging.ERROR)   # silence bare-mode warnings in scripts
    ok = {"streamlit_webrtc": False, "mediapipe": False, "tensorflow": False}
    if webrtc:
        try:
            import streamlit_webrtc  # noqa: F401
            ok["streamlit_webrtc"] = True
        except Exception:
            pass
    try:
        import mediapipe  # noqa: F401
        ok["mediapipe"] = True
    except Exception:
        pass
    if need_tf:
        try:
            import tensorflow  # noqa: F401
            ok["tensorflow"] = True
        except Exception:
            pass
    return ok
