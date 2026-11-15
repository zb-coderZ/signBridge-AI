"""Run MediaPipe ONCE over a dataset and save landmarks (never train on raw pixels).

Output: ``data/processed/<source>_<LANG>_<kind>.npz`` (see core/dataset.py for the layout).

Examples
--------
    python scripts/extract_landmarks.py --name asl_alphabet --max-per-class 400
    python scripts/extract_landmarks.py --path data/raw/wlasl --lang ASL --kind dynamic --source wlasl --max-per-class 30
Signer ids: pass --signer-regex with one capture group to read the signer from file names
(e.g. 'sub(\\d+)'); otherwise every sample gets signer 'unknown' and the split falls back to a random,
NOT signer-independent split (the report says so).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from core.config import load_config, path_of
from core.dataset import LandmarkSet, iter_sign_mnist
from core.dataset_info import DATASETS, IMG_EXT, VID_EXT, instructions
from core.hand_detector import DetectorUnavailable, HandDetector
from core.landmarks import observations_to_slots, resample_sequence

log = logging.getLogger("extract")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", choices=list(DATASETS)); ap.add_argument("--path"); ap.add_argument("--lang"); ap.add_argument("--kind", choices=["static", "dynamic"])
    ap.add_argument("--source"); ap.add_argument("--max-per-class", type=int, default=0); ap.add_argument("--signer-regex")
    args = ap.parse_args()
    cfg = load_config()
    if args.name == "sign_mnist":
        root, lang, kind, source = path_of("data_raw"), args.lang or "ASL", args.kind or "static", args.source or args.name
    elif args.name:
        d = DATASETS[args.name]
        root, lang, kind, source = path_of("data_raw") / d["folder"], args.lang or d["lang"], args.kind or d["kind"], args.source or args.name
    elif args.path and args.lang and args.kind:
        root, lang, kind, source = Path(args.path), args.lang.upper(), args.kind, args.source or Path(args.path).name
    else:
        ap.error("give --name, or --path with --lang and --kind")
    if not root.exists():
        print(f"Dataset folder {root} not found.\n" + (instructions(args.name) if args.name else ""))
        return 1
    try:
        det = HandDetector("image")
    except DetectorUnavailable as exc:
        print(exc); return 2
    T, mirror = cfg["sequence_length"], cfg["hand_detector"]["mirror_input"]
    exts = IMG_EXT if kind == "static" else VID_EXT
    sig_re = re.compile(args.signer_regex) if args.signer_regex else None
    raws, masks, ys, sids = [], [], [], []
    seen = skipped = 0

    def detect(bgr):
        bgr = cv2.flip(bgr, 1) if mirror else bgr            # same convention as the live app
        return observations_to_slots(det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    if args.name == "sign_mnist":
        csv_paths = [root / "sign_mnist_train.csv"]
        if not csv_paths[0].exists():
            csv_paths = [root / "sign_mnist_train" / "sign_mnist_train.csv"]
        if not csv_paths[0].exists():
            print(f"Sign Language MNIST training CSV not found in {root}.")
            return 1
        per_class: dict[str, int] = {}
        for label, gray in iter_sign_mnist(csv_paths[0]):
            if args.max_per_class and per_class.get(label, 0) >= args.max_per_class:
                continue
            per_class[label] = per_class.get(label, 0) + 1
            seen += 1
            img = cv2.cvtColor(cv2.resize(gray, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
            r, m = detect(img)
            if not m.any():
                skipped += 1
                continue
            raws.append(r)
            masks.append(m)
            ys.append(label)
            sids.append("sign_mnist_unknown")
    else:
        for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(f for f in cdir.rglob("*") if f.suffix.lower() in exts)
            if args.max_per_class:
                files = files[: args.max_per_class]
            for f in files:
                seen += 1
                sid = (sig_re.search(f.name).group(1) if sig_re and sig_re.search(f.name) else "unknown")
                if kind == "static":
                    img = cv2.imread(str(f))
                    if img is None:
                        skipped += 1; continue
                    r, m = detect(img)
                    if not m.any():
                        skipped += 1; continue
                else:
                    cap = cv2.VideoCapture(str(f)); frames = []
                    while True:
                        ok, img = cap.read()
                        if not ok:
                            break
                        frames.append(detect(img))
                    cap.release()
                    if sum(int(m.any()) for _, m in frames) < T // 3:
                        skipped += 1; continue
                    r, m = resample_sequence(frames, T)
                raws.append(r); masks.append(m); ys.append(cdir.name); sids.append(f"{source}_{sid}")
            log.info("class %-12s kept so far: %d", cdir.name, len(ys))
    if not ys:
        print("No hands detected in any file - nothing saved."); return 3
    out = path_of("data_processed") / f"{source}_{lang}_{kind}.npz"
    LandmarkSet(np.stack(raws), np.stack(masks), np.array(ys), np.array(sids), lang, kind, {"source": source}).save(out)
    print(f"Saved {len(ys)} samples ({skipped}/{seen} skipped: unreadable or no hand) -> {out}")
    if len(set(sids)) < 3:
        print("NOTE: fewer than 3 signer ids -> training will use a random split that is NOT signer-independent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
