"""Evaluate the trained CNN and Random Forest on the official real test CSV."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from core.compat import ensure_import_order

ensure_import_order(need_tf=True, webrtc=False)

import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from core.config import path_of
from core.image_dataset import find_dataset, load_csv
from core.image_models import ImageModel


def main() -> int:
    paths = find_dataset(path_of("image_data"))
    x, y_raw = load_csv(paths.test)
    model = ImageModel(path_of("image_models"))
    if not model.available:
        print("No trained model found. Run python scripts/train_image.py first.")
        return 1
    labels = model.labels
    index = {label: i for i, label in enumerate(labels)}
    y = np.asarray([index[label] for label in y_raw])
    results = {}
    if model.cnn is not None:
        pred = np.argmax(model.cnn.predict(x, verbose=0), axis=1)
        results["cnn"] = {"accuracy": float(accuracy_score(y, pred)),
                          "macro_f1": float(f1_score(y, pred, average="macro"))}
    if model.rf is not None:
        pred = model.rf.predict(x.reshape(len(x), -1))
        results["random_forest"] = {"accuracy": float(accuracy_score(y, pred)),
                                    "macro_f1": float(f1_score(y, pred, average="macro"))}
    out = path_of("reports") / "evaluation.json"
    out.write_text(json.dumps({"dataset_test": str(paths.test), "n_test": len(y), "models": results}, indent=2),
                   encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
