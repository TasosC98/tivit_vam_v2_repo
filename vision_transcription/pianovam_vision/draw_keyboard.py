"""Render the rectified keyboard strip for recordings, with per-key borders and
note-name labels, to visually verify the corner mapping for each video.

White-key separators are drawn in red, black keys outlined in cyan, and every
key is labelled with its note name (white keys with octave, e.g. 'C4'; black
keys as 'C#', 'D#', ...). Optionally overlays the pitches the TSV says are
pressed at ``--time`` (green) as an extra sanity check.

    # one recording:
    python -m pianovam_vision.draw_keyboard --config configs/default.yaml \
        --record_time 2024-02-14_19-10-09 --time 30 --out_dir preview_keys/

    # ALL recordings (skips data.exclude_records):
    python -m pianovam_vision.draw_keyboard --config configs/default.yaml \
        --out_dir preview_keys/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .keyboard import (
    is_white_key, key_geometry, note_name, perspective_matrix, pitch_to_x,
    warp_frame, white_key_pitches,
)
from .labels import read_tsv
from .metadata import load_recordings


def annotate(strip_bgr, width: int, height: int, active=None) -> None:
    """Draw key borders + note labels (and optional active pitches) in place."""
    import cv2

    wp = white_key_pitches()
    n = len(wp)
    wk = width / n
    boxes = key_geometry(width, height)
    black_h = next(y1 for (k, _, _, _, y1) in boxes.values() if k == "black")
    red, cyan, green = (0, 0, 255), (255, 255, 0), (0, 255, 0)

    # White-key separators. Where a black key sits on the boundary, only draw the
    # lower part (so the black key photo stays clear); at E-F / B-C gaps draw full.
    for b in range(0, n + 1):
        x = min(int(round(b * wk)), width - 1)
        full = b == 0 or b == n or (1 <= b <= n - 1 and wp[b] - wp[b - 1] == 1)
        cv2.line(strip_bgr, (x, 0 if full else black_h), (x, height), red, 1)

    # Black keys (outlined) + labels for every key.
    for p, (kind, x0, y0, x1, y1) in boxes.items():
        if kind == "black":
            cv2.rectangle(strip_bgr, (x0, 0), (x1, y1), cyan, 1)
            cv2.putText(strip_bgr, note_name(p)[:-1], (x0 + 1, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, cyan, 1, cv2.LINE_AA)
        else:
            cx = (x0 + x1) // 2
            cv2.putText(strip_bgr, note_name(p), (max(0, cx - 11), height - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, red, 1, cv2.LINE_AA)

    for p in (active or []):
        x = pitch_to_x(p, width)
        cv2.line(strip_bgr, (x, 0), (x, height), green, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record_time", default=None,
                    help="omit to render ALL recordings")
    ap.add_argument("--time", type=float, default=20.0,
                    help="timestamp (s) of the frame to grab")
    ap.add_argument("--width", type=int, default=1872, help="output strip width")
    ap.add_argument("--height", type=int, default=240, help="output strip height")
    ap.add_argument("--show_active", action="store_true",
                    help="also overlay TSV-active pitches at --time (green)")
    ap.add_argument("--out_dir", default="preview_keys")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    import cv2
    import decord

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
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
        vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        try:
            vr = decord.VideoReader(str(vp), num_threads=1)
            fps = float(vr.get_avg_fps()) or 60.0
            fidx = max(0, min(int(round(args.time * fps)), len(vr) - 1))
            frame = vr[fidx].asnumpy()                      # (H,W,3) RGB

            mat = perspective_matrix(r.corners, W, H)
            strip = warp_frame(frame, mat, W, H, grayscale=False)
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)

            active = None
            if args.show_active:
                notes = read_tsv(r.tsv_path(root, cfg["data"]["tsv_dir"]),
                                 cfg["labels"]["offset_field"])
                t = args.time
                active = [n.pitch for n in notes
                          if n.onset - 0.05 <= t <= n.offset + 0.05]
            annotate(strip_bgr, W, H, active)

            fn = out / f"{r.record_time}_keys.png"
            cv2.imwrite(str(fn), strip_bgr)
            ok += 1
            print(f"[{ok + fail}/{len(recs)}] wrote {fn.name}")
        except Exception as e:                              # corrupt video -> skip
            fail += 1
            print(f"[{ok + fail}/{len(recs)}] SKIP {r.record_time}: {e}")

    print(f"done: {ok} images written, {fail} skipped -> {out}/")


if __name__ == "__main__":
    main()
