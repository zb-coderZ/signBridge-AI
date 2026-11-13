#!/usr/bin/env bash
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source venv/bin/activate
exec streamlit run app/streamlit_app.py "$@"
