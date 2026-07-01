"""Sanity-check tool: verify keyboard corners, warping, and label sync.

Saves two PNGs for a chosen frame:
  *_raw.png    : original frame with the keyboard quad drawn
  *_warp.png   : rectified strip with vertical markers at the pitches that are
                 active (per the TSV) at that timestamp -> if the markers line
                 up with visibly pressed keys, corners + sync are correct.

    python -m pianovam_vision.preview --config configs/default.yaml \
        --record_time 2024-02-14_19-10-09 --time 12.0 --out_dir preview/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .keyboard import perspective_matrix, pitch_to_column, warp_frame
from .labels import read_tsv
from .metadata import index_by_record_time, load_recordings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record_time", required=True)
    ap.add_argument("--time", type=float, default=10.0, help="timestamp in seconds")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="a note counts as active within +/- this many seconds")
    ap.add_argument("--snap_to_onset", action="store_true",
                    help="jump to the nearest note onset (best for verifying sync)")
    ap.add_argument("--out_dir", default="preview")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    import cv2
    import decord

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
    recs = index_by_record_time(load_recordings(root / cfg["data"]["metadata"]))
    rec = recs[args.record_time]

    vp = rec.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
    vr = decord.VideoReader(str(vp), num_threads=1)
    native_fps = float(vr.get_avg_fps()) or 60.0

    # Notes first, so --snap_to_onset can pick a moment that definitely has a key down.
    notes = read_tsv(rec.tsv_path(root, cfg["data"]["tsv_dir"]), cfg["labels"]["offset_field"])
    t = args.time
    if args.snap_to_onset and notes:
        t = min(notes, key=lambda n: abs(n.onset - args.time)).onset
        print(f"snapped to nearest onset: {t:.3f}s")

    fidx = int(round(t * native_fps))
    fidx = max(0, min(fidx, len(vr) - 1))
    frame = vr[fidx].asnumpy()  # (H,W,3) RGB

    kb = cfg["keyboard"]
    w, h = kb["warp_width"], kb["warp_height"]
    mat = perspective_matrix(rec.corners, w, h)
    strip = warp_frame(frame, mat, w, h, grayscale=False)

    # A note is "active" if its [onset, offset] overlaps [t-tol, t+tol]. The
    # tolerance matters because key-press durations here are only ~50-80 ms.
    tol = args.tolerance
    active = [n.pitch for n in notes if n.onset - tol <= t <= n.offset + tol]
    nearest = sorted(notes, key=lambda n: abs(n.onset - t))[:8]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    quad = rec.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(raw, [quad], True, (0, 0, 255), 3)
    cv2.imwrite(str(out / f"{args.record_time}_{t:.2f}_raw.png"), raw)

    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    for p in active:
        x = pitch_to_column(p, w)
        cv2.line(strip_bgr, (x, 0), (x, h), (0, 255, 0), 1)
    cv2.imwrite(str(out / f"{args.record_time}_{t:.2f}_warp.png"), strip_bgr)

    print(f"active pitches @ {t:.2f}s (+/-{tol}s): {sorted(active)}")
    print("nearest notes (onset, offset, pitch):")
    for n in nearest:
        print(f"  {n.onset:8.3f}  {n.offset:8.3f}  pitch={n.pitch}")
    print(f"wrote PNGs to {out}/")


if __name__ == "__main__":
    main()
