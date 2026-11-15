"""Streaming and validation helpers for the Sign Language MNIST CSV files."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
    23: "X", 24: "Y",
}
LETTER_TO_LABEL = {letter: label for label, letter in LABEL_TO_LETTER.items()}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train: Path
    test: Path


def _find_one(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name), key=lambda p: (len(p.relative_to(root).parts), str(p).lower()))
    return matches[0] if matches else None


def find_dataset(root: str | Path) -> DatasetPaths:
    """Find the shallowest train/test CSVs and reject incomplete datasets."""
    root = Path(root)
    for candidate in (root, root.parent):
        train, test = _find_one(candidate, "sign_mnist_train.csv"), _find_one(candidate, "sign_mnist_test.csv")
        if train is not None and test is not None:
            return DatasetPaths(candidate, train, test)
    raise FileNotFoundError(
        f"Sign Language MNIST CSVs not found below {root}. Place sign_mnist_train.csv and "
        "sign_mnist_test.csv in data/raw/sign_language_mnist/."
    )


def _validate_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        fields = next(csv.reader(fh), [])
    pixels = [name for name in fields[1:] if name.startswith("pixel")]
    if not fields or fields[0] != "label" or len(pixels) != 784:
        raise ValueError(f"{path} must contain label plus 784 pixel columns")
    return fields


def load_csv(path: str | Path, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV into normalised ``(N, 28, 28, 1)`` images and integer labels."""
    path = Path(path)
    fields = _validate_header(path)
    pixels = fields[1:]
    images: list[np.ndarray] = []
    labels: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, fieldnames=fields)
        next(reader)
        for row in reader:
            label = int(row["label"])
            if label not in LABEL_TO_LETTER:
                raise ValueError(f"Unexpected label {label} in {path}; expected the 24 ASL MNIST labels")
            labels.append(label)
            images.append(np.asarray([int(row[name]) for name in pixels], dtype=np.float32).reshape(28, 28))
            if limit and len(labels) >= limit:
                break
    if not images:
        raise ValueError(f"{path} contains no samples")
    x = np.asarray(images, dtype=np.float32)[..., None] / 255.0
    return x, np.asarray(labels, dtype=np.int64)


def train_validation_split(
    x: np.ndarray, y: np.ndarray, validation_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible stratified split for CNN training."""
    return train_test_split(x, y, test_size=validation_size, random_state=seed, stratify=y)

