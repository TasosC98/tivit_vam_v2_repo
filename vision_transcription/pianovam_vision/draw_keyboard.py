"""Render the rectified keyboard strip for recordings, with per-key borders and
note-name labels, to visually verify the corner mapping for each video.

White-key separators are drawn in red, black keys outlined in cyan, and every
key is labelled with its note name (white keys with octave, e.g. 'C4'; black
keys as 'C#', 'D#', ...). With ``--show_active`` it also draws the pitches the
TSV says are pressed at ``--time`` (green lines) AND writes their note names in
green in a header band above the strip. ``--busiest`` picks the timestamp with
the most simultaneous notes (best for checking that chords/polyphony line up).

    # one recording at a chord (peak polyphony):
    python -m pianovam_vision.draw_keyboard --config configs/default.yaml \
        --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_keys/

    # ALL recordings at a fixed time:
    python -m pianovam_vision.draw_keyboard --config configs/default.yaml \
        --time 20 --out_dir preview_keys/

    # ALL PianoYT recordings at their biggest chord STRIKE, 10 per contact sheet.
    # --onsets_only matters for PianoYT: its audio-derived labels keep pedal-
    # sustained keys "active", so at peak polyphony many green lines would sit on
    # released keys even with a perfect crop.
    python -m pianovam_vision.draw_keyboard --config configs/pianoyt.yaml \
        --busiest --onsets_only --sheet 10 --out_dir preview_keys_yt/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .keyboard import (
    is_white_key, key_geometry, note_name, perspective_matrix, pitch_to_x,
    warp_frame, white_key_pitches,
)
from .labels import read_reference
from .metadata import recordings_from_cfg


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


def add_active_header(strip_bgr, width: int, active, band_h: int = 48):
    """Return a taller image with a black header band naming each pressed key in
    green above its column (two staggered rows so chord notes don't overlap)."""
    import cv2

    band = np.zeros((band_h, width, 3), dtype=strip_bgr.dtype)
    green = (0, 255, 0)
    rows = [18, 40]
    for i, p in enumerate(sorted(active)):
        x = pitch_to_x(p, width)
        label = note_name(p)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = int(min(max(0, x - tw // 2), width - tw))
        y = rows[i % 2]
        cv2.putText(band, label, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    green, 1, cv2.LINE_AA)
        cv2.line(band, (x, y + 3), (x, band_h), green, 1)  # tick to the strip
    return np.vstack([band, strip_bgr])


def label_bar(text: str, width: int, band_h: int = 28):
    """Black band with white caption, stacked above each strip in a sheet."""
    import cv2

    band = np.zeros((band_h, width, 3), dtype=np.uint8)
    cv2.putText(band, text, (6, band_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return band


def write_sheet(parts, out: Path, index: int) -> None:
    import cv2

    fn = out / f"sheet_{index:03d}.png"
    cv2.imwrite(str(fn), np.vstack(parts))
    print(f"  -> contact sheet {fn.name} ({len(parts) // 2} strips)")


STRUCK_WINDOW = 0.10   # --onsets_only: keys struck within this many s before t


def peak_onset_time(notes, default: float) -> float:
    """Timestamp just after the largest chord strike (most onsets within
    STRUCK_WINDOW). Unlike peak polyphony, this ignores notes that are only
    sounding because of the sustain pedal (PianoYT labels come from audio)."""
    ons = sorted(n.onset for n in notes)
    if not ons:
        return default
    best, best_t, j = 0, default, 0
    for i, t in enumerate(ons):
        while ons[j] < t - STRUCK_WINDOW:
            j += 1
        if i - j + 1 > best:
            best, best_t = i - j + 1, t
    return best_t + 0.02   # a moment after the last strike: all keys are down


def peak_polyphony_time(notes, default: float) -> float:
    """Timestamp where the most notes are simultaneously held."""
    if not notes:
        return default
    events = sorted([(n.onset, 1) for n in notes] + [(n.offset, -1) for n in notes])
    cur = best = 0
    best_t = default
    for t, d in events:
        cur += d
        if cur > best:
            best, best_t = cur, t
    return best_t + 0.005  # nudge so the onset notes are counted as held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record_time", default=None,
                    help="omit to render ALL recordings")
    ap.add_argument("--time", type=float, default=20.0,
                    help="timestamp (s) of the frame to grab")
    ap.add_argument("--busiest", action="store_true",
                    help="use the moment with the most simultaneous notes "
                         "(implies --show_active)")
    ap.add_argument("--width", type=int, default=1872, help="output strip width")
    ap.add_argument("--height", type=int, default=240, help="output strip height")
    ap.add_argument("--show_active", action="store_true",
                    help="overlay TSV-active pitches (green lines + names)")
    ap.add_argument("--onsets_only", action="store_true",
                    help="mark only keys STRUCK in the last 100 ms (and with --busiest "
                         "pick the biggest chord strike). Use for PianoYT, whose "
                         "audio-derived labels keep pedal-sustained keys 'active'")
    ap.add_argument("--sheet", type=int, default=0,
                    help="also stack N strips per contact-sheet image (0 = off)")
    ap.add_argument("--out_dir", default="preview_keys")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    import cv2
    import decord

    show_active = args.show_active or args.busiest

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
    recs = recordings_from_cfg(cfg)
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
    sheet_parts, sheet_idx = [], 0

    for r in recs:
        vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        try:
            notes = None
            t = args.time
            if show_active:
                notes = read_reference(r, cfg)
                if args.busiest:
                    t = (peak_onset_time if args.onsets_only
                         else peak_polyphony_time)(notes, args.time)

            vr = decord.VideoReader(str(vp), num_threads=1)
            fps = float(vr.get_avg_fps()) or 60.0
            fidx = max(0, min(int(round(t * fps)), len(vr) - 1))
            frame = vr[fidx].asnumpy()                      # (H,W,3) RGB

            mat = perspective_matrix(r.corners, W, H)
            strip = warp_frame(frame, mat, W, H, grayscale=False)
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)

            active = None
            if show_active and notes is not None:
                if args.onsets_only:
                    active = sorted(n.pitch for n in notes
                                    if t - STRUCK_WINDOW <= n.onset <= t)
                else:
                    active = sorted(n.pitch for n in notes
                                    if n.onset - 0.05 <= t <= n.offset + 0.05)
            annotate(strip_bgr, W, H, active)
            if active:
                strip_bgr = add_active_header(strip_bgr, W, active)

            fn = out / f"{r.record_time}_keys.png"
            cv2.imwrite(str(fn), strip_bgr)
            ok += 1
            extra = ""
            if active is not None:
                extra = (f"  t={t:.2f}s  {len(active)} pressed: "
                         f"{[note_name(p) for p in active]}")
            print(f"[{ok + fail}/{len(recs)}] wrote {fn.name}{extra}")
            if args.sheet > 0:
                nh, nw = frame.shape[:2]
                what = "keys just struck" if args.onsets_only else "labelled keys down"
                caption = (f"{r.record_time}  [{r.split}]  {nw}x{nh}  t={t:.2f}s"
                           + (f"  {len(active)} {what}" if active else ""))
                if strip_bgr.shape[0] == H:       # no active-key header: pad to align
                    strip_bgr = np.vstack([np.zeros((48, W, 3), np.uint8), strip_bgr])
                sheet_parts += [label_bar(caption, W), strip_bgr]
                if len(sheet_parts) // 2 >= args.sheet:
                    write_sheet(sheet_parts, out, sheet_idx)
                    sheet_parts, sheet_idx = [], sheet_idx + 1
        except Exception as e:                              # corrupt video -> skip
            fail += 1
            print(f"[{ok + fail}/{len(recs)}] SKIP {r.record_time}: {e}")

    if sheet_parts:
        write_sheet(sheet_parts, out, sheet_idx)
    print(f"done: {ok} images written, {fail} skipped -> {out}/")


if __name__ == "__main__":
    main()
