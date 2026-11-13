"""Evaluate models on the signer-held-out test split and write everything to reports/.

Outputs (in ``reports/`` or ``reports/demo/``): ``metrics.json``, ``results_table.md`` and one
confusion-matrix PNG per (language, kind, model). Every number comes from this run; nothing is
hard-coded. Latency/FPS are measured on this machine (CPU) and exclude camera capture.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import mediapipe  # noqa: F401

import argparse
import logging
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from core.config import ROOT, load_config, path_of, resolve
from core.dataset import load_many
from core.engine import RecognitionEngine
from core.models_io import load_json, load_language_models, save_json
from core.training import make_features, three_way_split

log = logging.getLogger("evaluate")


def plot_confusion(cm: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * n + 3), max(4, 0.5 * n + 2.5)))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes,
                cbar_kws={"label": "row-normalised"}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140); plt.close(fig)


def per_sample_latency(pred, feats: np.ndarray, n: int = 100) -> float:
    """Mean single-sample model latency in ms (features already built)."""
    idx = np.linspace(0, len(feats) - 1, min(n, len(feats))).astype(int)
    pred.predict_proba(feats[idx[0]])                           # warm-up
    t0 = time.perf_counter()
    for i in idx:
        pred.predict_proba(feats[i])
    return (time.perf_counter() - t0) / len(idx) * 1000


def pipeline_latency(models_dir: Path, lang: str, ds_static, ds_dynamic) -> dict:
    """End-to-end per-frame latency of the engine (features -> models -> stability), detector excluded."""
    lm = load_language_models(lang, [models_dir])
    eng = RecognitionEngine({lang: lm})
    eng.router.set_mode(lang)
    out = {}
    for name, ds, kind in (("static", ds_static, "Static only"), ("dynamic", ds_dynamic, "Dynamic only")):
        if ds is None or getattr(lm, name) is None:
            continue
        eng.settings.sign_mode = kind
        eng.clear_all()
        frames = []
        for i in range(min(len(ds), 40)):
            r, m = ds.raw[i], ds.mask[i]
            frames += [(r[t], m[t]) for t in range(r.shape[0])] if r.ndim == 4 else [(r, m)]
        times, t = [], 0.0
        for r, m in frames[:600]:
            t0 = time.perf_counter(); eng.process_landmarks(r, m, t); times.append((time.perf_counter() - t0) * 1000); t += 1 / 30
        times = np.array(times[5:])
        out[name] = {"mean_ms": float(times.mean()), "p95_ms": float(np.percentile(times, 95)), "fps_pipeline_only": float(1000 / times.mean()), "frames": int(len(times))}
    return out


def detector_latency() -> dict:
    """Time the MediaPipe hand landmarker on a blank frame if the model is available."""
    try:
        from core.hand_detector import HandDetector
        det = HandDetector("image")
    except Exception as exc:
        return {"available": False, "reason": str(exc).splitlines()[0]}
    img = np.zeros((480, 640, 3), np.uint8)
    det.detect(img)
    t0 = time.perf_counter()
    for _ in range(30):
        det.detect(img)
    ms = (time.perf_counter() - t0) / 30 * 1000
    return {"available": True, "mean_ms_blank_640x480": ms, "note": "blank frame (no hand): lower bound, see README"}


def evaluate_component(kind: str, lang: str, models_dir: Path, report_dir: Path, tag: str) -> tuple[dict | None, object]:
    cfg = load_config()["training"]
    meta_p = models_dir / f"{kind}_{lang}_meta.json"
    if not meta_p.exists():
        return None, None
    meta = load_json(meta_p)
    classes = load_json(models_dir / f"{kind}_{lang}_labels.json")["classes"]
    ds = load_many([resolve(f) for f in meta["data_files"]])
    tr, va, te, mode = three_way_split(ds, meta["seed"], cfg["test_size"], cfg["val_size"])
    test = ds.subset(te)
    X = make_features(test, kind)
    cls = {c: i for i, c in enumerate(classes)}
    y = np.array([cls[v] for v in test.y])
    lm = load_language_models(lang, [models_dir])
    comp = getattr(lm, kind)
    res = {"is_demo": meta.get("is_demo", False), "split_mode": mode, "n_test": int(len(te)),
           "test_signers": sorted(set(test.signer.tolist())), "n_train_signers": int(len(set(ds.signer[tr].tolist()))),
           "classes": classes, "models": {}}
    for name, pred in comp.predictors.items():
        probs = pred.predict_proba_batch(X)
        yp = probs.argmax(1)
        p, r, f, s = precision_recall_fscore_support(y, yp, labels=list(range(len(classes))), zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
        cm = confusion_matrix(y, yp, labels=list(range(len(classes))))
        png = report_dir / f"confusion_{kind}_{lang}_{name}.png"
        plot_confusion(cm, classes, f"{lang} {kind} {name}  (test acc {accuracy_score(y, yp):.3f}, {mode})"
                       + ("  [SYNTHETIC DEMO DATA]" if res["is_demo"] else ""), png)
        res["models"][name] = {
            "accuracy": float(accuracy_score(y, yp)), "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(), "confusion_png": str(png.name),
            "latency_ms_per_sample_model_only": per_sample_latency(pred, X)}
    return res, test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="evaluate models/demo and write to reports/demo")
    ap.add_argument("--models-dir"); ap.add_argument("--report-dir"); ap.add_argument("--lang", default="all")
    args = ap.parse_args()
    cfg = load_config()
    models_dir = Path(args.models_dir) if args.models_dir else path_of("models_demo" if args.demo else "models_trained")
    report_dir = Path(args.report_dir) if args.report_dir else path_of("reports") / ("demo" if args.demo else "")
    langs = cfg["languages"] if args.lang.lower() == "all" else [args.lang.upper()]
    out = {"environment": {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
                           "note": "latency measured on CPU, camera capture excluded"},
           "models_dir": (lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p))(models_dir.resolve()), "languages": {}}
    for lang in langs:
        entry, tests = {}, {}
        for kind in ("static", "dynamic"):
            res, test = evaluate_component(kind, lang, models_dir, report_dir, "")
            if res:
                entry[kind] = res; tests[kind] = test
                for name, m in res["models"].items():
                    log.info("[%s/%s/%s] acc=%.4f macroF1=%.4f (%s, n=%d)", lang, kind, name, m["accuracy"], m["macro_f1"], res["split_mode"], res["n_test"])
        if entry:
            entry["pipeline_latency"] = pipeline_latency(models_dir, lang, tests.get("static"), tests.get("dynamic"))
            out["languages"][lang] = entry
    if not out["languages"]:
        log.error("No trained models found in %s - run scripts/build_demo.py or the train_* scripts first.", models_dir)
        return 1
    out["hand_detector_latency"] = detector_latency()
    save_json(report_dir / "metrics.json", out)
    rows = ["| Language | Type | Model | Test accuracy | Macro-F1 | Model latency (ms/sample) | Split | n_test |", "|---|---|---|---|---|---|---|---|"]
    for lang, e in out["languages"].items():
        for kind in ("static", "dynamic"):
            if kind in e:
                for name, m in e[kind]["models"].items():
                    rows.append(f"| {lang} | {kind} | {name.upper()} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['latency_ms_per_sample_model_only']:.2f} | "
                                f"{e[kind]['split_mode']}{' (SYNTHETIC)' if e[kind]['is_demo'] else ''} | {e[kind]['n_test']} |")
    fps = ["", "| Language | Type | Pipeline mean (ms/frame) | p95 (ms) | FPS (pipeline only) |", "|---|---|---|---|---|"]
    for lang, e in out["languages"].items():
        for kind, v in e.get("pipeline_latency", {}).items():
            fps.append(f"| {lang} | {kind} | {v['mean_ms']:.2f} | {v['p95_ms']:.2f} | {v['fps_pipeline_only']:.0f} |")
    (report_dir / "results_table.md").write_text("\n".join(rows + fps) + "\n", encoding="utf-8")
    log.info("Reports written to %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
