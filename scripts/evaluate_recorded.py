"""Evaluate recorded webcam images with the trained CNN and report domain shift."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from core.compat import ensure_import_order

ensure_import_order(need_tf=True, webrtc=False)

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from core.config import path_of, resolve
from core.image_dataset import LABEL_TO_LETTER
from core.image_engine import ImageRecognitionEngine
from core.image_models import ImageModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="data/recorded")
    args = ap.parse_args()
    root = Path(args.path)
    if not root.is_absolute():
        root = resolve(root)
    model = ImageModel(path_of("image_models"))
    engine = ImageRecognitionEngine(model)
    true, pred = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        if not folder.name.isalpha():
            continue
        for path in folder.glob("*.*"):
            image = cv2.imread(str(path))
            result = engine.process(image)
            if result.prediction:
                true.append(folder.name.upper())
                pred.append(result.prediction)
    engine.close()
    if not true:
        print(f"No usable recorded images found in {root}.")
        return 1
    score = {"n_samples": len(true), "accuracy": float(accuracy_score(true, pred)),
             "report": classification_report(true, pred, output_dict=True, zero_division=0),
             "note": "This score is a separate webcam/domain-shift evaluation, not a replacement for the official test score."}
    out = path_of("reports") / "recorded_metrics.json"
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(json.dumps({"n_samples": score["n_samples"], "accuracy": score["accuracy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
