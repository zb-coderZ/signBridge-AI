"""Record real webcam images for a separate domain-shift evaluation set.

Controls: SPACE captures a frame, Q or ESC quits. Images are never used as
training data automatically; run ``scripts/evaluate_recorded.py`` explicitly.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import cv2

from core.config import resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="ASL MNIST letter, for example A or B")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--output", default="data/recorded")
    args = ap.parse_args()
    label = args.label.strip().upper()
    if len(label) != 1 or label in {"J", "Z"} or not label.isalpha():
        ap.error("label must be one ASL MNIST letter; J and Z are not in the dataset")
    output = Path(args.output)
    if not output.is_absolute():
        output = resolve(output)
    target = output / label
    target.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open webcam #{args.camera}. Close other apps or try --camera 1.")
        return 2
    saved = 0
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Label {label} | {saved}/{args.count} | SPACE capture Q quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        cv2.imshow("SignBridge recorder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == 32:
            path = target / f"{label}_{int(time.time() * 1000)}.jpg"
            if cv2.imwrite(str(path), frame):
                saved += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
