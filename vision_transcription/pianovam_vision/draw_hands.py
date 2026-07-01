"""Verify the derived per-note hand labels visually.

Warps a frame, draws the key borders, then colours each PRESSED key by the hand
the skeleton assigns to it (blue = Left, red = Right) and overlays that hand's
fingertips. Use it to eyeball whether the hand assignment matches the video.

  # busiest moment of one recording:
  python -m pianovam_vision.draw_hands --config configs/tiled_best.yaml \
      --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_hands/

  # a fixed time for ALL recordings:
  python -m pianovam_vision.draw_hands --config configs/tiled_best.yaml \
      --time 30 --out_dir preview_hands/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import hands as H
from .config import load_config
from .draw_keyboard import annotate, peak_polyphony_time
from .keyboard import note_name, perspective_matrix, pitch_to_x, warp_frame
from .labels import read_tsv
from .metadata import load_recordings

LEFT_BGR = (255, 60, 0)     # blue
RIGHT_BGR = (0, 60, 255)    # red
HAND_BGR = {"Left": LEFT_BGR, "Right": RIGHT_BGR, "unknown": (0, 255, 0)}


def render(strip_bgr, width, height, active, hand_of, tips):
    import cv2

    # Colour each pressed key's column + a header label by its assigned hand.
    band_h = 48
    band = np.zeros((band_h, width, 3), dtype=strip_bgr.dtype)
    rows = [18, 40]
    for i, p in enumerate(sorted(active)):
        hand = hand_of.get(p, "unknown")
        col = HAND_BGR[hand]
        x = pitch_to_x(p, width)
        cv2.line(strip_bgr, (x, 0), (x, height), col, 2)
        label = f"{hand[0]}:{note_name(p)}"          # e.g. "L:C3" / "R:E4"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = int(min(max(0, x - tw // 2), width - tw))
        y = rows[i % 2]
        cv2.putText(band, label, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
        cv2.line(band, (x, y + 3), (x, band_h), col, 1)

    # Overlay each detected hand's fingertips (so you can see the fingers).
    for hand, pts in tips.items():
        for (fx, fy) in pts:
            if 0 <= fx < width and 0 <= fy < height:
                cv2.circle(strip_bgr, (int(fx), int(fy)), 4, HAND_BGR[hand], -1)
    return np.vstack([band, strip_bgr])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record_time", default=None, help="omit to render ALL")
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--busiest", action="store_true",
                    help="use the moment with the most simultaneous notes")
    ap.add_argument("--width", type=int, default=1872)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--out_dir", default="preview_hands")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    import cv2
    import decord

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
    skel_dir = cfg["data"].get("handskeleton_dir", "Handskeleton")
    recs = load_recordings(root / cfg["data"]["metadata"])
    excl = set(cfg["data"].get("exclude_records", []) or [])
    if args.record_time:
        recs = [r for r in recs if r.record_time == args.record_time]
        if not recs:
            raise SystemExit(f"record_time {args.record_time!r} not found")
    else:
        recs = [r for r in recs if r.record_time not in excl]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, H = args.width, args.height
    ok = fail = 0

    for r in recs:
        try:
            sk = H.load_skeleton(H.skeleton_path(root, r.record_time, skel_dir))
            notes = read_tsv(r.tsv_path(root, cfg["data"]["tsv_dir"]),
                             cfg["labels"]["offset_field"])

            vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
            vr = decord.VideoReader(str(vp), num_threads=1)
            fps = float(vr.get_avg_fps()) or 60.0
            t = peak_polyphony_time(notes, args.time) if args.busiest else args.time
            fidx = max(0, min(int(round(t * fps)), len(vr) - 1))
            frame = vr[fidx].asnumpy()
            nh, nw = frame.shape[:2]

            mat = perspective_matrix(r.corners, W, H)
            strip = warp_frame(frame, mat, W, H, grayscale=False)
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
            annotate(strip_bgr, W, H)                      # key borders + labels

            active_notes = [n for n in notes if n.onset - 0.05 <= t <= n.offset + 0.05]
            active = sorted(n.pitch for n in active_notes)
            hands = H.assign_hands(active_notes, sk, r.corners, W, H, nw, nh, fps)
            hand_of = {n.pitch: h for n, h in zip(active_notes, hands)}
            tips = H.hands_at_frame(sk, fidx, r.corners, W, H, nw, nh)

            img = render(strip_bgr, W, H, active, hand_of, tips)
            fn = out / f"{r.record_time}_hands.png"
            cv2.imwrite(str(fn), img)
            ok += 1
            lab = [f"{h[0]}:{note_name(p)}" for p, h in
                   sorted(hand_of.items())]
            print(f"[{ok + fail}/{len(recs)}] {fn.name}  t={t:.2f}s  {lab}")
        except Exception as e:
            fail += 1
            print(f"[{ok + fail}/{len(recs)}] SKIP {r.record_time}: {e}")

    print(f"done: {ok} images, {fail} skipped -> {out}/")


if __name__ == "__main__":
    main()
