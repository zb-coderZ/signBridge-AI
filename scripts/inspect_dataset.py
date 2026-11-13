"""Scan a dataset folder: class counts, image/video counts, imbalance and a sample preview.

Examples
--------
    python scripts/inspect_dataset.py --name asl_alphabet
    python scripts/inspect_dataset.py --path D:/data/my_signs
    python scripts/inspect_dataset.py --all          # checks every known dataset, prints download help if missing
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from core.config import path_of
from core.dataset_info import DATASETS, IMG_EXT, VID_EXT, instructions

log = logging.getLogger("inspect_dataset")


def scan(root: Path) -> dict:
    """Return per-class image/video counts for ``root/<class>/<files>`` (recursive inside a class)."""
    classes: dict[str, dict] = {}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [f for f in cdir.rglob("*") if f.suffix.lower() in IMG_EXT]
        vids = [f for f in cdir.rglob("*") if f.suffix.lower() in VID_EXT]
        if imgs or vids:
            classes[cdir.name] = {"images": len(imgs), "videos": len(vids), "sample": str((imgs or vids)[0])}
    wl = next(iter(root.glob("WLASL*.json")), None)
    return {"classes": classes, "wlasl_json": str(wl) if wl else None}


def preview(info: dict, out: Path, n: int = 12) -> bool:
    tiles = []
    for name, c in list(info["classes"].items())[:n]:
        p = Path(c["sample"])
        if p.suffix.lower() in IMG_EXT:
            img = cv2.imread(str(p))
        else:
            cap = cv2.VideoCapture(str(p)); ok, img = cap.read(); cap.release()
            img = img if ok else None
        if img is None:
            continue
        img = cv2.resize(img, (160, 160))
        cv2.putText(img, name[:14], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        return False
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return True


def report(label: str, root: Path) -> dict | None:
    if not root.exists() or not any(root.iterdir()):
        print(f"\n== {label}: NOT FOUND at {root}")
        if label in DATASETS:
            print(instructions(label))
        return None
    info = scan(root)
    cl = info["classes"]
    if not cl:
        print(f"\n== {label}: folder exists but no class sub-folders with images/videos were found in {root}")
        return None
    counts = {k: v["images"] + v["videos"] for k, v in cl.items()}
    vals = np.array(list(counts.values()))
    imb = float(vals.max() / max(vals.min(), 1))
    print(f"\n== {label}: {root}")
    print(f"classes: {len(cl)} | images: {sum(v['images'] for v in cl.values())} | videos: {sum(v['videos'] for v in cl.values())}")
    print(f"per-class min/median/max: {vals.min()}/{int(np.median(vals))}/{vals.max()} | imbalance ratio (max/min): {imb:.2f}"
          + ("  <-- imbalanced, consider class weights / more samples" if imb > 3 else ""))
    for k, v in Counter(counts).most_common(0):
        pass
    for k in sorted(counts, key=counts.get)[:5]:
        print(f"  fewest: {k}: {counts[k]}")
    if info["wlasl_json"]:
        data = json.load(open(info["wlasl_json"]))
        print(f"WLASL annotation json: {len(data)} glosses, {sum(len(g['instances']) for g in data)} instances "
              "(many video URLs are dead - only count what exists on disk)")
    png = path_of("reports") / "figures" / f"preview_{label}.png"
    if preview(info, png):
        print(f"sample preview saved to {png}")
    return {"classes": len(cl), "counts": counts, "imbalance": imb}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", choices=list(DATASETS)); ap.add_argument("--path"); ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    results = {}
    if args.path:
        results[Path(args.path).name] = report(Path(args.path).name, Path(args.path))
    names = list(DATASETS) if args.all or not (args.name or args.path) else ([args.name] if args.name else [])
    for n in names:
        results[n] = report(n, path_of("data_raw") / DATASETS[n]["folder"])
    found = [k for k, v in results.items() if v]
    print(f"\nSummary: {len(found)} dataset(s) found {found}; missing ones are skipped - the project still runs with the demo model "
          "and with your own recordings (scripts/record_samples.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
