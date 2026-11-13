"""Train static-sign models (Random Forest baseline + Keras MLP), one set per language.

Examples
--------
    python scripts/train_static.py --lang ASL                 # uses data/processed/*_ASL_static.npz (not demo_*)
    python scripts/train_static.py --lang all --demo          # synthetic demo data -> models/demo
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (must be first)
import mediapipe  # noqa: F401  (mediapipe BEFORE tensorflow/keras)

import argparse
import datetime as dt
import glob
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from core.config import load_config, path_of
from core.dataset import load_many
from core.models_io import save_json
from core.training import (augmented_features, build_mlp, compute_static_references, make_features,
                           three_way_split)

from core.config import ROOT


def rel(f) -> str:
    """Store data paths relative to the project root so models/metadata are portable."""
    try:
        return Path(f).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(f)


log = logging.getLogger("train_static")


def find_data(lang: str, demo: bool) -> list[str]:
    files = sorted(glob.glob(str(path_of("data_processed") / f"*_{lang}_static.npz")))
    return [f for f in files if Path(f).name.startswith("demo_") == demo]


def train_language(lang: str, files: list[str], out_dir: Path, report_dir: Path, demo: bool) -> dict:
    import keras
    cfg = load_config()
    tc, seed = cfg["training"], cfg["training"]["seed"]
    keras.utils.set_random_seed(seed)
    ds = load_many(files)
    classes = sorted(set(ds.y.tolist()))
    tr, va, te, mode = three_way_split(ds, seed, tc["test_size"], tc["val_size"])
    log.info("[%s] %d samples, %d classes, split=%s train/val/test=%d/%d/%d", lang, len(ds), len(classes), mode,
             len(tr), len(va), len(te))
    if mode != "cross_signer":
        log.warning("[%s] fewer than 3 signers: split is NOT signer-independent", lang)
    Xtr, ytr = augmented_features(ds, tr, "static", classes, seed)
    cls = {c: i for i, c in enumerate(classes)}
    Xva = make_features(ds.subset(va), "static")
    yva = np.array([cls[v] for v in ds.y[va]])

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=tc["rf_estimators"], class_weight="balanced", n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    rf_val = float(accuracy_score(yva, rf.predict(Xva)))
    log.info("[%s] RF val acc %.4f (%.1fs)", lang, rf_val, time.time() - t0)

    t0 = time.time()
    mlp = build_mlp(Xtr.shape[1], len(classes))
    hist = mlp.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=tc["mlp_epochs"], batch_size=tc["batch_size"], verbose=0,
                   callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")])
    mlp_val = float(accuracy_score(yva, mlp.predict(Xva, verbose=0).argmax(1)))
    log.info("[%s] MLP val acc %.4f (%d epochs, %.1fs)", lang, mlp_val, len(hist.history["loss"]), time.time() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_dir / f"static_{lang}_rf.joblib")
    mlp.save(out_dir / f"static_{lang}_mlp.keras")
    save_json(out_dir / f"static_{lang}_labels.json", {"classes": classes})
    refs = compute_static_references(ds, classes)
    ref_path = out_dir / f"references_{lang}.npz"
    old = dict(np.load(ref_path, allow_pickle=False)) if ref_path.exists() else {}
    np.savez(ref_path, **{**old, **refs})
    meta = {"is_demo": demo, "models": ["rf", "mlp"], "default": "mlp" if mlp_val >= rf_val else "rf",
            "val_accuracy": {"rf": rf_val, "mlp": mlp_val}, "split_mode": mode, "seed": seed,
            "n_samples": len(ds), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "n_signers": int(len(set(ds.signer.tolist()))), "data_files": [rel(f) for f in files],
            "created": dt.datetime.now().isoformat(timespec="seconds"), "classes": classes}
    save_json(out_dir / f"static_{lang}_meta.json", meta)
    save_json(report_dir / f"train_static_{lang}.json", {**meta, "mlp_history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}})
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all", help="PSL | ASL | all")
    ap.add_argument("--data", nargs="*", help="landmark .npz files (default: data/processed/*_<LANG>_static.npz)")
    ap.add_argument("--demo", action="store_true", help="use demo_* synthetic data and write to models/demo")
    ap.add_argument("--out-dir", help="override the models output directory")
    args = ap.parse_args()
    cfg = load_config()
    langs = cfg["languages"] if args.lang.lower() == "all" else [args.lang.upper()]
    out_dir = Path(args.out_dir) if args.out_dir else path_of("models_demo" if args.demo else "models_trained")
    report_dir = path_of("reports") / ("demo" if args.demo else "")
    done = 0
    for lang in langs:
        files = args.data or find_data(lang, args.demo)
        if not files:
            log.error("[%s] no static landmark files found. Extract landmarks (scripts/extract_landmarks.py), "
                      "record your own (scripts/record_samples.py) or run scripts/build_demo.py.", lang)
            continue
        train_language(lang, files, out_dir, report_dir, args.demo)
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
