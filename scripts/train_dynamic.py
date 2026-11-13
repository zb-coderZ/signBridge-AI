"""Train dynamic-sign models: LSTM vs GRU ablation (same data, same split, same hyper-parameters).

Examples
--------
    python scripts/train_dynamic.py --lang ASL
    python scripts/train_dynamic.py --lang all --demo
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import mediapipe  # noqa: F401  (mediapipe BEFORE tensorflow/keras)

import argparse
import datetime as dt
import glob
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from core.config import load_config, path_of
from core.dataset import load_many
from core.models_io import save_json
from core.training import (augmented_features, build_rnn, compute_dynamic_prototypes, make_features,
                           three_way_split)

from core.config import ROOT


def rel(f) -> str:
    """Store data paths relative to the project root so models/metadata are portable."""
    try:
        return Path(f).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(f)


log = logging.getLogger("train_dynamic")


def find_data(lang: str, demo: bool) -> list[str]:
    files = sorted(glob.glob(str(path_of("data_processed") / f"*_{lang}_dynamic.npz")))
    return [f for f in files if Path(f).name.startswith("demo_") == demo]


def train_language(lang: str, files: list[str], out_dir: Path, report_dir: Path, demo: bool) -> dict:
    import keras
    cfg = load_config()
    tc, seed, T = cfg["training"], cfg["training"]["seed"], cfg["sequence_length"]
    ds = load_many(files)
    classes = sorted(set(ds.y.tolist()))
    tr, va, te, mode = three_way_split(ds, seed, tc["test_size"], tc["val_size"])
    log.info("[%s] %d sequences, %d classes, split=%s train/val/test=%d/%d/%d", lang, len(ds), len(classes), mode,
             len(tr), len(va), len(te))
    Xtr, ytr = augmented_features(ds, tr, "dynamic", classes, seed)
    cls = {c: i for i, c in enumerate(classes)}
    Xva = make_features(ds.subset(va), "dynamic")
    yva = np.array([cls[v] for v in ds.y[va]])

    out_dir.mkdir(parents=True, exist_ok=True)
    val_acc, hist_all, secs = {}, {}, {}
    for kind in ("lstm", "gru"):                       # ablation: identical data / split / seed / hyper-parameters
        keras.utils.set_random_seed(seed)
        model = build_rnn(kind, T, Xtr.shape[2], len(classes), tc["rnn_units"])
        t0 = time.time()
        h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=tc["rnn_epochs"], batch_size=tc["batch_size"], verbose=0,
                      callbacks=[keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-4),
                                 keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")])
        secs[kind] = time.time() - t0
        val_acc[kind] = float(accuracy_score(yva, model.predict(Xva, verbose=0).argmax(1)))
        hist_all[kind] = {k: [float(v) for v in vs] for k, vs in h.history.items()}
        model.save(out_dir / f"dynamic_{lang}_{kind}.keras")
        log.info("[%s] %s val acc %.4f (%d epochs, %.1fs, %d params)", lang, kind.upper(), val_acc[kind],
                 len(h.history["loss"]), secs[kind], model.count_params())
    save_json(out_dir / f"dynamic_{lang}_labels.json", {"classes": classes})
    refs = compute_dynamic_prototypes(ds, classes)
    ref_path = out_dir / f"references_{lang}.npz"
    old = dict(np.load(ref_path, allow_pickle=False)) if ref_path.exists() else {}
    np.savez(ref_path, **{**old, **refs})
    meta = {"is_demo": demo, "models": ["lstm", "gru"], "default": max(val_acc, key=val_acc.get),
            "val_accuracy": val_acc, "train_seconds": secs, "split_mode": mode, "seed": seed, "n_samples": len(ds),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_signers": int(len(set(ds.signer.tolist()))),
            "data_files": [rel(f) for f in files], "created": dt.datetime.now().isoformat(timespec="seconds"),
            "classes": classes}
    save_json(out_dir / f"dynamic_{lang}_meta.json", meta)
    save_json(report_dir / f"train_dynamic_{lang}.json", {**meta, "history": hist_all})
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all")
    ap.add_argument("--data", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    cfg = load_config()
    langs = cfg["languages"] if args.lang.lower() == "all" else [args.lang.upper()]
    out_dir = Path(args.out_dir) if args.out_dir else path_of("models_demo" if args.demo else "models_trained")
    report_dir = path_of("reports") / ("demo" if args.demo else "")
    done = 0
    for lang in langs:
        files = args.data or find_data(lang, args.demo)
        if not files:
            log.error("[%s] no dynamic landmark files found. Extract from videos (scripts/extract_landmarks.py --kind dynamic), "
                      "record your own (scripts/record_samples.py --kind dynamic) or run scripts/build_demo.py.", lang)
            continue
        train_language(lang, files, out_dir, report_dir, args.demo)
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
