"""Download instructions for each supported dataset (printed when a dataset is missing)."""
DATASETS = {
    "asl_alphabet": {
        "lang": "ASL", "kind": "static", "folder": "asl_alphabet",
        "layout": "<folder>/<CLASS>/*.jpg  (e.g. asl_alphabet_train/A/A1.jpg)",
        "how": "Kaggle 'ASL Alphabet' (grassknoted/asl-alphabet). Needs a free Kaggle account:\n"
               "  pip install kaggle ; kaggle datasets download -d grassknoted/asl-alphabet -p data/raw/asl_alphabet --unzip\n"
               "  or download in the browser: https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
    },
    "wlasl": {
        "lang": "ASL", "kind": "dynamic", "folder": "wlasl",
        "layout": "<folder>/<gloss>/*.mp4  (or the official WLASL_v0.3.json + videos/ folder)",
        "how": "WLASL (Li et al., WACV 2020): https://github.com/dxli94/WLASL\n"
               "  Clone the repo, use start_kit/video_downloader.py to fetch the videos. MANY YouTube links are dead:\n"
               "  use only the subset that downloads (WLASL100 is plenty). Arrange videos as data/raw/wlasl/<gloss>/<id>.mp4.",
    },
    "msasl": {
        "lang": "ASL", "kind": "dynamic", "folder": "msasl",
        "layout": "<folder>/<gloss>/*.mp4",
        "how": "MS-ASL (optional): https://www.microsoft.com/en-us/download/details.aspx?id=100121\n"
               "  Download the JSON annotation files and fetch/clip the videos yourself (links are partly dead).",
    },
    "psl": {
        "lang": "PSL", "kind": "static", "folder": "psl",
        "layout": "<folder>/<CLASS>/*.jpg (static letters)  and/or <folder>/dynamic/<gloss>/*.mp4",
        "how": "Pakistan Sign Language: the LUMS / CHISEL PSL data and the Dynamic PSL datasets may require\n"
               "  contacting the authors for access (see their papers / project pages, or Kaggle 'Pakistan Sign Language' uploads).\n"
               "  If you cannot obtain them, RECORD YOUR OWN: python scripts/record_samples.py --lang PSL --label <X> --kind static",
    },
    "sign_mnist": {
        "lang": "ASL", "kind": "static", "folder": ".",
        "layout": "sign_mnist_train.csv and sign_mnist_test.csv (28x28 grayscale rows)",
        "how": "Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist\n"
               "  The repository accepts the CSV files directly from data/raw; labels cover 24 ASL letters (J and Z are absent).",
    },
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CSV_EXT = {".csv"}


def instructions(name: str) -> str:
    d = DATASETS[name]
    return f"[{name}] expected at data/raw/{d['folder']}  layout: {d['layout']}\n{d['how']}"
