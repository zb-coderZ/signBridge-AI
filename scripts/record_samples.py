"""Record your own samples with the webcam (fallback when PSL data is unavailable; also adds signers).

Controls (OpenCV window):  SPACE = record   Q / ESC = quit.
    static : hold the sign still; SPACE captures --frames frames (default 40) as individual samples.
    dynamic: press SPACE, a 3-2-1 countdown runs, then 30 frames (~1 s) are captured as ONE sample.

Examples
--------
    python scripts/record_samples.py --lang PSL --label alif --kind static --signer zb
    python scripts/record_samples.py --lang ASL --label HELLO --kind dynamic --signer zb --count 20
Samples are appended to data/processed/custom_<LANG>_<kind>.npz. Use a different --signer name
for every person so cross-signer evaluation is meaningful. Record an 'unknown' label (random
in-between hand poses) to help the model reject non-signs.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import logging
import time

import cv2
import numpy as np

from core.config import load_config, path_of
from core.dataset import LandmarkSet, concat
from core.hand_detector import DetectorUnavailable, HandDetector, draw_hands
from core.landmarks import observations_to_slots

log = logging.getLogger("record")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["PSL", "ASL"]); ap.add_argument("--label", required=True)
    ap.add_argument("--kind", choices=["static", "dynamic"], default="static"); ap.add_argument("--signer", default="me")
    ap.add_argument("--frames", type=int, default=40); ap.add_argument("--count", type=int, default=10, help="dynamic: sequences to record")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    T = cfg["sequence_length"]
    try:
        det = HandDetector("video")
    except DetectorUnavailable as exc:
        print(exc); return 2
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"Cannot open webcam #{a.camera}. Close other apps using it, or try --camera 1."); return 3
    raws, masks = [], []
    state, t_start, buf = "idle", 0.0, []
    signer = f"custom_{a.signer}"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed."); break
        frame = cv2.flip(frame, 1) if cfg["hand_detector"]["mirror_input"] else frame
        hands = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(time.monotonic() * 1000))
        raw, mask = observations_to_slots(hands)
        draw_hands(frame, hands)
        msg = f"{a.lang} '{a.label}' {a.kind} | saved: {len(raws)} | SPACE=record Q=quit"
        if state == "countdown":
            left = 3 - int(time.monotonic() - t_start)
            if left <= 0:
                state, buf = "recording", []
            else:
                msg = f"Get ready... {left}"
        elif state == "recording":
            if a.kind == "static":
                if mask.any():
                    raws.append(raw); masks.append(mask); buf.append(1)
                msg = f"Recording {len(buf)}/{a.frames}"
                if len(buf) >= a.frames:
                    state = "idle"
            else:
                buf.append((raw, mask)); msg = f"Recording {len(buf)}/{T}"
                if len(buf) >= T:
                    if sum(int(m.any()) for _, m in buf) >= T // 2:
                        raws.append(np.stack([b[0] for b in buf])); masks.append(np.stack([b[1] for b in buf]))
                    else:
                        print("Hand missing in too many frames - sample discarded.")
                    state = "idle"
                    if len(raws) >= a.count:
                        break
        cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SignBridge recorder", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == 32 and state == "idle":
            state, t_start = ("countdown" if a.kind == "dynamic" else "recording"), time.monotonic()
            buf = []
    cap.release(); cv2.destroyAllWindows()
    if not raws:
        print("Nothing recorded."); return 0
    new = LandmarkSet(np.stack(raws), np.stack(masks), np.array([a.label] * len(raws)), np.array([signer] * len(raws)), a.lang, a.kind,
                      {"source": "custom"})
    out = path_of("data_processed") / f"custom_{a.lang}_{a.kind}.npz"
    final = concat([LandmarkSet.load(out), new]) if out.exists() else new
    final.meta = {"source": "custom"}
    final.save(out)
    print(f"Saved {len(new)} new samples (total {len(final)}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
