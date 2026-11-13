"""Make ``core`` importable when a script is run as ``python scripts/<name>.py`` and fix import order."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compat import ensure_import_order  # noqa: E402  (sets TF log env, imports webrtc->mediapipe->tf)
from core.config import setup_logging  # noqa: E402

ensure_import_order(webrtc=False)  # webrtc is only needed by the Streamlit app
setup_logging()
