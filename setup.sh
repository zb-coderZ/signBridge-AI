#!/usr/bin/env bash
# SignBridge AI setup (Linux / macOS). Tries Python 3.14, then 3.12 / 3.11 / 3.13, and keeps the first one
# on which mediapipe + tensorflow + the other packages install AND import.
set -u
cd "$(dirname "$0")"
for PY in python3.14 python3.12 python3.11 python3.13 python3; do
  command -v "$PY" >/dev/null 2>&1 || continue
  echo "=== Trying $($PY --version 2>&1) ($PY) ==="
  rm -rf venv
  "$PY" -m venv venv || continue
  # shellcheck disable=SC1091
  source venv/bin/activate
  python -m pip install --upgrade pip -q
  if pip install -r requirements.txt && python scripts/check_env.py; then
    echo "$($PY --version 2>&1)" > venv/PYTHON_USED.txt
    echo "=== Environment OK with $($PY --version 2>&1) ==="
    if [ ! -f models/demo/static_ASL_meta.json ]; then echo "Building demo models..."; python scripts/build_demo.py; fi
    echo "Done. Start the app with: ./run_app.sh"
    exit 0
  fi
  echo "!!! $PY failed (mediapipe/tensorflow wheels often lag behind new Python versions). Trying the next interpreter..."
  deactivate 2>/dev/null || true
done
echo "No suitable Python found. Install Python 3.12 (https://www.python.org/downloads/) and re-run ./setup.sh"
exit 1
