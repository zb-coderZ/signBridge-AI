from pathlib import Path

import numpy as np

from core.dataset import SIGN_MNIST_LABELS, iter_sign_mnist


def test_iter_sign_mnist_decodes_letter_and_image(tmp_path: Path):
    header = ",".join(["label"] + [f"pixel{i}" for i in range(1, 785)])
    row = ",".join(["0"] + ["7"] * 784)
    path = tmp_path / "sign_mnist.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    label, image = next(iter_sign_mnist(path))

    assert label == "A"
    assert image.shape == (28, 28)
    assert image.dtype == np.uint8
    assert np.all(image == 7)
    assert len(SIGN_MNIST_LABELS) == 24
