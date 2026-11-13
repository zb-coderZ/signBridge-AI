"""Verify that the native libraries import (in the safe order) on THIS interpreter. Exit 0 = OK."""
import importlib
import sys

ORDER = ["streamlit_webrtc", "mediapipe", "tensorflow", "cv2", "numpy", "pandas", "sklearn", "streamlit", "pyttsx3", "speech_recognition"]
bad = []
print(f"Python {sys.version.split()[0]}")
for name in ORDER:                      # streamlit_webrtc MUST come before mediapipe/tensorflow
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:20s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  FAIL {name:20s} {type(exc).__name__}: {exc}")
        bad.append(name)
try:
    from mediapipe.tasks.python import vision  # noqa: F401
    print("  OK   mediapipe Tasks HandLandmarker API")
except Exception as exc:
    print(f"  FAIL mediapipe Tasks API: {exc}")
    bad.append("mediapipe.tasks")
sys.exit(1 if bad else 0)
