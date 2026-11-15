from pathlib import Path

import numpy as np

from core.image_dataset import LABEL_TO_LETTER, find_dataset, load_csv


def _write_csv(path: Path, label: int) -> None:
    fields = ["label"] + [f"pixel{i}" for i in range(1, 785)]
    path.write_text(",".join(fields) + "\n" + ",".join([str(label)] + ["255"] * 784) + "\n", encoding="utf-8")


def test_find_dataset_prefers_shallow_duplicate(tmp_path: Path):
    root = tmp_path / "sign_language_mnist"
    (root / "nested").mkdir(parents=True)
    _write_csv(root / "sign_mnist_train.csv", 0)
    _write_csv(root / "sign_mnist_test.csv", 0)
    _write_csv(root / "nested" / "sign_mnist_train.csv", 0)
    _write_csv(root / "nested" / "sign_mnist_test.csv", 0)

    paths = find_dataset(root)
    assert paths.train == root / "sign_mnist_train.csv"
    assert paths.test == root / "sign_mnist_test.csv"


def test_load_csv_returns_normalized_image(tmp_path: Path):
    path = tmp_path / "sign_mnist_train.csv"
    _write_csv(path, 0)

    images, labels = load_csv(path)

    assert images.shape == (1, 28, 28, 1)
    assert images.dtype == np.float32
    assert images.min() == 1.0
    assert labels.tolist() == [0]
    assert LABEL_TO_LETTER[0] == "A"
