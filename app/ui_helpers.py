"""Small UI helpers: CSS, charts, tables."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
.sb-hero {background: linear-gradient(90deg,#0f766e,#2563eb); color:#fff; padding:14px 20px; border-radius:12px; margin-bottom:10px;}
.sb-hero h1 {margin:0; font-size:1.7rem;} .sb-hero p {margin:2px 0 0 0; opacity:.9;}
.sb-demo {background:#fff7ed; border:1px solid #fdba74; color:#9a3412; padding:8px 12px; border-radius:8px; font-size:.9rem;}
.sb-sentence {font-size:1.6rem; font-weight:600; background:#f1f5f9; border-radius:10px; padding:12px 16px; min-height:3.2rem; color:#0f172a;}
.sb-pred {font-size:2.4rem; font-weight:700; line-height:1.1; color:#0f172a;}
@media (prefers-color-scheme: dark){ .sb-sentence{background:#1e293b;color:#f8fafc;} .sb-pred{color:#f8fafc;} }
</style>
"""


def hero() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="sb-hero"><h1>SignBridge AI</h1><p>Sign language &rarr; text &amp; speech, and speech &rarr; sign, '
                'for Pakistan Sign Language (PSL) and American Sign Language (ASL)</p></div>', unsafe_allow_html=True)


def demo_banner(models: dict) -> None:
    demo = [l for l, m in models.items() if m.is_demo]
    if demo:
        st.markdown('<div class="sb-demo"><b>DEMO MODEL</b> active for ' + ", ".join(demo) + ': trained on <i>synthetic</i> landmark data. '
                    'Its classes are coarse finger patterns, <b>not real ASL/PSL signs</b>. Train on real data to replace it (see README).</div>',
                    unsafe_allow_html=True)


def top3_bars(top: list[tuple[str, float]]) -> None:
    for label, p in top:
        st.progress(min(max(p, 0.0), 1.0), text=f"{label}  {p * 100:.1f}%")


def conf_chart(conf: list) -> None:
    if not conf:
        st.caption("Confidence chart appears once a hand is detected.")
        return
    t0 = conf[0][0]
    df = pd.DataFrame({"seconds": [round(t - t0, 2) for t, _, _ in conf], "confidence": [c for _, c, _ in conf]}).set_index("seconds")
    st.line_chart(df, height=180)


def history_table(history: list[dict]) -> pd.DataFrame:
    cols = ["timestamp", "language", "type", "sign", "confidence"]
    return pd.DataFrame(history, columns=cols) if history else pd.DataFrame(columns=cols)


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
