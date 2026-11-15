"""Create real-data EDA, PCA/t-SNE, clustering, importance, saliency, and outlier reports."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from core.compat import ensure_import_order

ensure_import_order(need_tf=True, webrtc=False)

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

from core.config import path_of
from core.image_dataset import LABEL_TO_LETTER, find_dataset, load_csv


def _scatter(points: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in sorted(np.unique(labels)):
        selected = labels == label
        ax.scatter(points[selected, 0], points[selected, 1], s=5, alpha=.45, label=LABEL_TO_LETTER[int(label)])
    ax.set_title(title)
    ax.legend(ncol=6, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    paths = find_dataset(path_of("image_data"))
    x, y = load_csv(paths.train)
    labels = sorted(np.unique(y).tolist())
    out = path_of("reports") / "mining"
    out.mkdir(parents=True, exist_ok=True)
    flat = x.reshape(len(x), -1)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    sample, sample_y = flat[sample_idx], y[sample_idx]
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = np.bincount(y, minlength=26)
    ax.bar([LABEL_TO_LETTER[i] for i in sorted(np.unique(y))], [counts[i] for i in sorted(np.unique(y))])
    ax.set_title("Sign Language MNIST class balance")
    fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=140); plt.close(fig)
    pca = PCA(n_components=2, random_state=42).fit_transform(sample)
    _scatter(pca, sample_y, out / "pca.png", "PCA of pixel features")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42, init="pca").fit_transform(sample)
    _scatter(tsne, sample_y, out / "tsne.png", "t-SNE of pixel features")
    clusters = KMeans(n_clusters=24, random_state=42, n_init=10, max_iter=300).fit_predict(sample)
    purity = 0.0
    for cluster in np.unique(clusters):
        members = sample_y[clusters == cluster]
        purity += np.bincount(members, minlength=26).max()
    purity /= len(sample_y)
    iso = IsolationForest(n_estimators=100, contamination=.02, random_state=42, n_jobs=-1).fit(sample)
    flagged = np.where(iso.predict(flat) == -1)[0][:25]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(np.hstack([x[i, ..., 0] for i in flagged]) if len(flagged) else np.zeros((28, 28)), cmap="gray")
    ax.set_title("Examples flagged as pixel outliers")
    ax.axis("off"); fig.tight_layout(); fig.savefig(out / "outliers.png", dpi=140); plt.close(fig)
    rf_path = path_of("image_models") / "image_rf.joblib"
    importance = {}
    if rf_path.exists():
        rf = joblib.load(rf_path)
        imp = rf.feature_importances_.reshape(28, 28)
        importance = {"max": float(imp.max()), "mean": float(imp.mean())}
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(imp, cmap="magma"); ax.set_title("Random Forest pixel importance")
        ax.axis("off"); fig.tight_layout(); fig.savefig(out / "rf_feature_importance.png", dpi=140); plt.close(fig)
    saliency = {}
    cnn_path = path_of("image_models") / "image_cnn.keras"
    if cnn_path.exists():
        import tensorflow as tf
        import keras
        cnn = keras.models.load_model(cnn_path, compile=False)
        sample_input = tf.convert_to_tensor(x[[sample_idx[0]]])
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            prediction = cnn(sample_input, training=False)
            target = tf.argmax(prediction[0])
            score = prediction[0, target]
        gradient = np.abs(tape.gradient(score, sample_input).numpy()[0, ..., 0])
        saliency = {"predicted_class": LABEL_TO_LETTER[int(labels[int(target)])], "max": float(gradient.max()),
                    "mean": float(gradient.mean())}
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(gradient, cmap="inferno")
        ax.set_title("CNN input-gradient saliency")
        ax.axis("off"); fig.tight_layout(); fig.savefig(out / "cnn_saliency.png", dpi=140); plt.close(fig)
    summary = {"n_samples": len(x), "n_classes": len(np.unique(y)), "class_counts": {LABEL_TO_LETTER[i]: int(counts[i]) for i in np.unique(y)},
               "kmeans_cluster_purity": float(purity), "outlier_count_reported": int(len(flagged)), "rf_importance": importance,
               "cnn_saliency": saliency,
               "notes": "PCA/t-SNE/K-means/outliers use a reproducible 3,000-row sample; all rows come from the supplied training CSV."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path_of("reports") / "mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
