"""Build the SYNTHETIC demo data + demo models so the app works immediately.

Runs: generate synthetic landmarks -> train static models -> train dynamic models -> evaluate.
The demo shapes are coarse finger-extension patterns, NOT real ASL/PSL signs.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import mediapipe  # noqa: F401

import argparse
import logging
import subprocess
import sys

from core.config import ROOT, load_config, path_of
from core.dataset import LandmarkSet
from core.synthetic import build_dynamic_dataset, build_static_dataset

log = logging.getLogger("build_demo")


def generate(seed: int) -> None:
    cfg = load_config()
    d, T = cfg["demo"], cfg["sequence_length"]
    out = path_of("data_processed")
    for lang in cfg["languages"]:
        raw, mask, y, sid = build_static_dataset(lang, d["n_signers"], d["static_per_class_per_signer"], seed)
        LandmarkSet(raw, mask, y, sid, lang, "static", {"source": "synthetic_demo"}).save(out / f"demo_{lang}_static.npz")
        raw, mask, y, sid = build_dynamic_dataset(lang, d["n_signers"], d["dynamic_per_class_per_signer"], seed, T)
        LandmarkSet(raw, mask, y, sid, lang, "dynamic", {"source": "synthetic_demo"}).save(out / f"demo_{lang}_dynamic.npz")
        log.info("[%s] synthetic data written (%d static, %d dynamic samples)", lang, len(y) if False else raw.shape[0], raw.shape[0])


def run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    log.info("$ %s", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=load_config()["training"]["seed"])
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    generate(args.seed)
    run("train_static.py", "--lang", "all", "--demo")
    run("train_dynamic.py", "--lang", "all", "--demo")
    if not args.skip_eval:
        run("evaluate.py", "--demo")
    log.info("Demo build complete. Start the app with run_app.sh / run_app.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
