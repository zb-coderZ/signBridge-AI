"""Inspect Sign Language MNIST CSVs, class balance, and sample images."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config import path_of, resolve
from core.image_dataset import LABEL_TO_LETTER, find_dataset, load_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="dataset folder; default is config paths.image_data")
    args = ap.parse_args()
    root = resolve(args.path) if args.path else path_of("image_data")
    try:
        paths = find_dataset(root)
        x, y = load_csv(paths.train)
        test_x, test_y = load_csv(paths.test)
    except (FileNotFoundError, ValueError) as exc:
        print(f"DATASET ERROR: {exc}")
        print("Fix: download Sign Language MNIST from Kaggle and place both CSV files in "
              "data/raw/sign_language_mnist/. This project does not download data.")
        return 1
    labels = sorted(np.unique(y).tolist())
    counts = {LABEL_TO_LETTER[label]: int(np.sum(y == label)) for label in labels}
    print(f"train CSV: {paths.train}")
    print(f"test CSV:  {paths.test}")
    print(f"rows: train={len(y)} test={len(test_y)}")
    print(f"classes ({len(labels)}): {', '.join(LABEL_TO_LETTER[label] for label in labels)}")
    print(f"class counts: {counts}")
    print(f"pixel mean={float(x.mean()):.4f}, std={float(x.std()):.4f}, min={float(x.min()):.1f}, max={float(x.max()):.1f}")
    out = path_of("reports") / "dataset_samples.png"
    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for ax, label in zip(axes.ravel(), labels):
        sample = x[np.where(y == label)[0][0], ..., 0]
        ax.imshow(sample, cmap="gray")
        ax.set_title(LABEL_TO_LETTER[label])
        ax.axis("off")
    for ax in axes.ravel()[len(labels):]:
        ax.axis("off")
    fig.suptitle("One real Sign Language MNIST sample per class")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"sample grid: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
