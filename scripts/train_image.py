"""Train the real Sign Language MNIST CNN and Random Forest baseline.

The official test CSV is never used for fitting. Use ``--limit`` only for a quick
pipeline check; reported project results should use the default full dataset.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from core.compat import ensure_import_order

ensure_import_order(need_tf=True, webrtc=False)

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.config import load_config, path_of
from core.image_dataset import LABEL_TO_LETTER, find_dataset, load_csv, train_validation_split
from core.image_models import build_cnn, save_labels

log = logging.getLogger("train_image")


def _save_confusion(cm: np.ndarray, labels: list[int], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=[LABEL_TO_LETTER[x] for x in labels],
           yticklabels=[LABEL_TO_LETTER[x] for x in labels], xlabel="Predicted", ylabel="True", title=title)
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, int(value), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], raw_labels: list[int]) -> dict:
    names = [LABEL_TO_LETTER[x] for x in raw_labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                            output_dict=True, zero_division=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="limit each CSV for a quick smoke test")
    ap.add_argument("--epochs", type=int, default=0, help="override configured CNN epochs")
    args = ap.parse_args()
    cfg = load_config()
    paths = find_dataset(path_of("image_data"))
    limit = args.limit or int(cfg["training"].get("image_max_train", 0))
    x_train, y_train_raw = load_csv(paths.train, limit)
    x_test, y_test_raw = load_csv(paths.test, int(cfg["training"].get("image_max_test", 0)))
    labels = sorted(np.unique(y_train_raw).tolist())
    if sorted(np.unique(y_test_raw).tolist()) != labels:
        raise ValueError("Train and test CSVs do not contain the same classes.")
    index = {label: i for i, label in enumerate(labels)}
    y_train = np.asarray([index[x] for x in y_train_raw], dtype=np.int64)
    y_test = np.asarray([index[x] for x in y_test_raw], dtype=np.int64)
    x_fit, x_val, y_fit, y_val = train_validation_split(x_train, y_train, cfg["training"]["image_validation_size"],
                                                         int(cfg["training"]["seed"]))
    out = path_of("image_models")
    report_dir = path_of("reports")
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_labels(out / "image_labels.json", labels)
    history: dict = {}

    import keras
    keras.utils.set_random_seed(int(cfg["training"]["seed"]))
    cnn = build_cnn(len(labels))
    fit = cnn.fit(x_fit, y_fit, validation_data=(x_val, y_val),
                  epochs=args.epochs or int(cfg["training"]["image_epochs"]),
                  batch_size=int(cfg["training"]["image_batch_size"]), verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    history = {key: [float(value) for value in values] for key, values in fit.history.items()}
    cnn.save(out / "image_cnn.keras")
    started = time.perf_counter()
    cnn_pred = np.argmax(cnn.predict(x_test, verbose=0), axis=1)
    cnn_metrics = _metrics(y_test, cnn_pred, list(range(len(labels))), labels)
    cnn_metrics["inference_ms_per_image"] = float((time.perf_counter() - started) * 1000 / max(1, len(x_test)))
    _save_confusion(confusion_matrix(y_test_raw, np.asarray([labels[x] for x in cnn_pred]), labels=labels),
                    labels, report_dir / "confusion_image_cnn.png", "CNN - Sign Language MNIST")

    rf = RandomForestClassifier(n_estimators=int(cfg["training"]["image_rf_estimators"]),
                                class_weight="balanced", n_jobs=-1, random_state=int(cfg["training"]["seed"]))
    rf.fit(x_train.reshape(len(x_train), -1), y_train)
    started = time.perf_counter()
    rf_pred = rf.predict(x_test.reshape(len(x_test), -1))
    rf_metrics = _metrics(y_test, rf_pred, list(range(len(labels))), labels)
    rf_metrics["inference_ms_per_image"] = float((time.perf_counter() - started) * 1000 / max(1, len(x_test)))
    joblib.dump(rf, out / "image_rf.joblib")
    _save_confusion(confusion_matrix(y_test_raw, np.asarray([labels[x] for x in rf_pred]), labels=labels),
                    labels, report_dir / "confusion_image_rf.png", "Random Forest - Sign Language MNIST")

    metrics = {
        "dataset": "Sign Language MNIST",
        "dataset_train": str(paths.train),
        "dataset_test": str(paths.test),
        "n_train": int(len(x_train)), "n_test": int(len(x_test)), "classes": [LABEL_TO_LETTER[x] for x in labels],
        "models": {"cnn": cnn_metrics, "random_forest": rf_metrics},
        "cnn_history": history,
        "honesty": "The official split may share signers with training; webcam performance can be lower because of domain shift.",
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (report_dir / "results_table.md").write_text(
        "| Model | Accuracy | Macro-F1 | Inference ms/image |\n|---|---:|---:|---:|\n"
        f"| CNN | {cnn_metrics['accuracy']:.4f} | {cnn_metrics['macro_f1']:.4f} | {cnn_metrics['inference_ms_per_image']:.3f} |\n"
        f"| Random Forest | {rf_metrics['accuracy']:.4f} | {rf_metrics['macro_f1']:.4f} | {rf_metrics['inference_ms_per_image']:.3f} |\n",
        encoding="utf-8")
    print(json.dumps({name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]}
                      for name, value in metrics["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
