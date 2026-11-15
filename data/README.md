# Dataset

Use only the already downloaded Sign Language MNIST data:

```text
data/
  raw/
    sign_language_mnist/
      sign_mnist_train.csv
      sign_mnist_test.csv
```

The CSV format is `label` plus 784 pixel columns. Labels 0-25 represent ASL
letters, but 9 (J) and 25 (Z) do not occur. The loader searches subfolders and
selects the shallowest copy when duplicate Kaggle files exist.

Run:

```bat
python scripts\inspect_dataset.py --path data\raw\sign_language_mnist
python scripts\train_image.py
python scripts\data_mining.py
```

No downloader is included. Do not put a second dataset into this project
without documenting its source and folder placement.
