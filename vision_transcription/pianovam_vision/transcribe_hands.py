"""Transcribe a PianoVAM recording and say, for every note, which hand (and finger) played it.

    # notes AND hands from a KeyWise-hands checkpoint (GPU server, strip cache):
    python -m pianovam_vision.transcribe_hands --config configs/vam_hands.yaml \
        --checkpoint runs/vam_hands/best.pt --record_time 2024-02-17_21-44-37 \
        --out_dir out/hands_demo decode.onset_threshold=0.9 decode.frame_threshold=0.7

    # notes from any transcription (or the reference), hands from the key-ROI rule (CPU):
    python -m pianovam_vision.transcribe_hands --config configs/vam_hands.yaml \
        --record_time 2024-02-17_21-44-37 --notes_midi out/E1s2_vam_full_test/2024-02-17_21-44-37.mid \
        --hands roi --out_dir out/hands_demo

Writes to ``--out_dir``:
  <rec>_hands.tsv   onset, offset, note, velocity, hand (L/R), finger (1-5), p_hand
  <rec>_hands.mid   two tracks, "Left hand" and "Right hand" (+ "Undecided")
  <rec>_hands.png   piano roll of --plot_seconds from --plot_start: left hand
                    blue, right hand orange, finger number on each note
  <rec>_frames.png  (with --frames N, needs the video) N video moments: the
                    rectified keyboard plus the area in front of it, each pressed
                    key outlined in its hand's colour with its finger number,
                    and MediaPipe's fingertips (dots) for a visual check
and prints the decisions at the end.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from . import hands as H
from .config import config_from_checkpoint, load_config
from .keyboard import note_name
from .labels import Note
from .metadata import index_by_record_time, recordings_from_cfg

COLORS = {"L": "#2b6cb0", "R": "#dd6b20", "": "#888888"}          # blue / orange / grey
BGR = {"L": (176, 108, 43), "R": (32, 107, 221), "": (136, 136, 136)}


def plot_roll(path: Path, notes: List[Note], cls: np.ndarray, t0: float, dur: float,
              title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib is not installed: no piano-roll image")
        return False
    sel = [i for i, n in enumerate(notes) if n.offset > t0 and n.onset < t0 + dur]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i in sel:
        n, c = notes[i], int(cls[i])
        h = H.class_hand(c)
        ax.add_patch(plt.Rectangle((n.onset, n.pitch - 0.4), max(0.02, n.offset - n.onset), 0.8,
                                   color=COLORS[h], alpha=0.85, lw=0))
        if c >= 0 and dur <= 30:
            ax.text(n.onset + 0.01, n.pitch, str(H.class_finger(c)), fontsize=6.5,
                    va="center", ha="left", color="white", fontweight="bold")
    ps = [notes[i].pitch for i in sel] or [60]
    lo, hi = min(ps) - 2, max(ps) + 2
    ax.set_xlim(t0, t0 + dur)
    ax.set_ylim(lo, hi)
    ticks = [p for p in range(lo, hi + 1) if p % 12 == 0]
    ax.set_yticks(ticks, [note_name(p) for p in ticks])
    ax.axhline(59.5, color="#bbbbbb", lw=0.6, ls=":")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch")
    ax.set_title(title, fontsize=10)
    ax.legend(handles=[Patch(color=COLORS["L"], label="left hand"),
                       Patch(color=COLORS["R"], label="right hand")],
              loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def frame_sheet(path: Path, video: Path, rec, notes: List[Note], cls: np.ndarray,
                track, fps: float, t0: float, dur: float, n_frames: int) -> bool:
    """A few video moments, rectified, with each pressed key and the fingertips drawn."""
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed: no frame sheet")
        return False
    if not video.exists():
        print(f"no video {video}: no frame sheet")
        return False
    W, KH, MARGIN = 1408, 172, 1.0           # keyboard ~8:1, plus its height again in front
    Ht = int(KH * (1 + MARGIN))
    m = H.homography(rec.corners, np.array([[0, 0], [W, 0], [W, KH], [0, KH]], float))
    cx, hw, ln = H.key_layout()
    # The busiest onsets of the window, in time order.
    groups = [g for g in H.chord_groups(notes, 0.034)
              if t0 <= notes[g[0]].onset < t0 + dur]
    groups = sorted(sorted(groups, key=len, reverse=True)[:n_frames], key=lambda g: notes[g[0]].onset)
    if not groups:
        return False
    cap = cv2.VideoCapture(str(video))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    tiles = []
    for g in groups:
        t = notes[g[0]].onset + 1.5 / fps                    # just after the strike
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * native_fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        img = cv2.warpPerspective(frame, m.astype(np.float64), (W, Ht), flags=cv2.INTER_LINEAR)
        for i in g:
            n, c = notes[i], int(cls[i])
            k = n.pitch - 21
            if not 0 <= k < 88:
                continue
            h = H.class_hand(c)
            x0, x1 = int((cx[k] - hw[k]) * W), int((cx[k] + hw[k]) * W)
            cv2.rectangle(img, (x0, 1), (x1, int(ln[k] * KH)), BGR[h], 3)
            cv2.putText(img, str(H.class_finger(c)) if c >= 0 else "?", (x0 + 2, int(ln[k] * KH) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR[h], 2, cv2.LINE_AA)
        if track is not None:
            f = min(int(round(t * fps)), len(track) - 1)
            for hi, h in enumerate("LR"):
                if track.present[f, hi]:
                    for j in H.FINGERTIPS:
                        x, y = track.pts[f, hi, j, 0] * W, track.pts[f, hi, j, 1] * KH
                        cv2.circle(img, (int(x), int(y)), 5, BGR[h], -1)
                        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)
        names = " ".join(f"{note_name(notes[i].pitch)}={H.class_hand(int(cls[i])) or '?'}"
                         f"{H.class_finger(int(cls[i])) or ''}" for i in sorted(g, key=lambda i: notes[i].pitch))
        band = np.full((30, W, 3), 255, np.uint8)
        cv2.putText(band, f"{t:7.2f} s   {names}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, img]))
    cap.release()
    if not tiles:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(tiles))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record_time", required=True)
    ap.add_argument("--checkpoint", default=None, help="KeyWise(-hands) checkpoint")
    ap.add_argument("--notes_midi", default=None, help="take the notes from this MIDI file")
    ap.add_argument("--reference", action="store_true", help="take the notes from the labels")
    ap.add_argument("--hands", default=None, choices=["model", "roi", "tip", "pitch"],
                    help="how to decide (default: model with a hand checkpoint, else roi)")
    ap.add_argument("--chord_matching", action="store_true")
    ap.add_argument("--out_dir", default="out/hands")
    ap.add_argument("--plot_start", type=float, default=None, help="s (default: first note)")
    ap.add_argument("--plot_seconds", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=4, help="video moments to draw (0 = none)")
    ap.add_argument("--show", type=int, default=20, help="decisions to print")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ckpt = None
    if args.checkpoint:
        import torch

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = config_from_checkpoint(ckpt, args.config, args.overrides)
    else:
        cfg = load_config(args.config, args.overrides)
    has_hand_model = bool(ckpt) and bool((cfg["model"].get("hands") or {}).get("enable", False))
    method = args.hands or ("model" if has_hand_model else "roi")
    if method == "model" and not has_hand_model:
        raise SystemExit("--hands model needs --checkpoint of a model with a hand branch")
    hc = cfg.get("hands") or {}
    fps, lf = cfg["labels"]["fps"], int(hc.get("label_frames", 8))
    rec = index_by_record_time(recordings_from_cfg(cfg))[args.record_time]

    track = None
    if method in ("roi", "tip") or (method == "model" and (cfg["model"]["hands"]).get("skeleton", True)) \
            or args.frames:
        try:
            track = H.load_skeleton(cfg, rec)
        except FileNotFoundError as e:
            if method != "pitch" and method != "model":
                raise
            print(f"no skeleton ({e})")

    probs = None
    if args.notes_midi:
        from .score_midi import load_midi_notes
        notes, source = load_midi_notes(Path(args.notes_midi)), Path(args.notes_midi).name
    elif args.reference:
        from .labels import read_reference
        notes, source = read_reference(rec, cfg), "reference labels"
    elif ckpt is not None:
        source = Path(args.checkpoint).parent.name
        notes = None
    else:
        raise SystemExit("where do the notes come from? --checkpoint, --notes_midi or --reference")

    if ckpt is not None and (notes is None or method == "model"):
        import torch

        from .decode import decode_with_cfg
        from .infer import predict_outputs
        from .model import build_model
        from .strip_cache import open_reader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        reader = open_reader(cfg, rec, args.max_frames)
        n = len(reader)
        skel = None
        if getattr(model, "hand_branch", None) is not None and model.hand_branch.skel is not None \
                and track is not None:
            skel = track.fit(n).model_input(np.arange(n))
        print(f"running {args.checkpoint} on {rec.record_time} ({n} frames, {reader.source})")
        o = predict_outputs(model, reader, cfg, device, skel)
        probs = o["hands"]
        if notes is None:
            notes = decode_with_cfg(o["onset"], o["frame"], o["velocity"], cfg)
    if args.max_frames:
        notes = [n for n in notes if n.onset < args.max_frames / fps]
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))

    if method == "model":
        scores = H.probs_scores(notes, probs, fps, lf)
    elif method == "pitch":
        scores = H.pitch_split(notes, int(hc.get("split_pitch", 60)))
    else:
        scores = H.fingertip_scores(notes, track, fps, lf, roi=(method == "roi"))
    cls, conf = H.decide(notes, scores, args.chord_matching or bool(hc.get("chord_matching", False)))

    out = Path(args.out_dir)
    H.write_hand_tsv(out / f"{rec.record_time}_hands.tsv", notes, cls, conf)
    try:
        H.write_hand_midi(out / f"{rec.record_time}_hands.mid", notes, cls)
        wrote_midi = True
    except ImportError:
        wrote_midi = False
    t0 = args.plot_start if args.plot_start is not None else (notes[0].onset if notes else 0.0)
    title = (f"{rec.record_time}: notes from {source}, hands by "
             f"{'KeyWise-hands' if method == 'model' else method}")
    plotted = plot_roll(out / f"{rec.record_time}_hands.png", notes, cls, t0, args.plot_seconds, title)
    sheet = False
    if args.frames:
        from .strip_cache import video_path_for
        sheet = frame_sheet(out / f"{rec.record_time}_frames.png", video_path_for(cfg, rec), rec,
                            notes, cls, track, fps, t0, args.plot_seconds, args.frames)

    # ---- the decisions
    hand = np.array([H.class_hand(int(c)) for c in cls])
    n_l, n_r, n_u = int((hand == "L").sum()), int((hand == "R").sum()), int((hand == "").sum())
    tot = max(1, len(notes))
    print(f"\n=== {rec.record_time}: {len(notes)} notes from {source}; hands by "
          f"{'KeyWise-hands' if method == 'model' else method} ===")
    print(f"  left hand  {n_l:6d} ({100 * n_l / tot:4.1f}%)   right hand {n_r:6d} "
          f"({100 * n_r / tot:4.1f}%)" + (f"   undecided {n_u}" if n_u else ""))
    if len(conf):
        print(f"  hand confidence: median {np.median(conf[cls >= 0]) if (cls >= 0).any() else 0:.2f}, "
              f"below 0.8 for {int(((conf < 0.8) & (cls >= 0)).sum())} notes")
    show = [i for i, n in enumerate(notes) if n.onset >= t0][:args.show]
    if show:
        print(f"\n  {'time (s)':>9}  {'note':>5}  {'hand':<6} {'finger':<11} {'p(hand)':>7}")
        for i in show:
            n, c = notes[i], int(cls[i])
            f = H.class_finger(c)
            print(f"  {n.onset:9.3f}  {note_name(n.pitch):>5}  "
                  f"{(H.HAND_NAMES.get(H.class_hand(c), 'undecided')):<6} "
                  f"{(f'{f} ({H.FINGER_NAMES[f]})' if f else '-'):<11} {conf[i]:7.2f}")
    print(f"\n  wrote {out / (rec.record_time + '_hands.tsv')}"
          + (f", {rec.record_time}_hands.mid" if wrote_midi else "")
          + (f", {rec.record_time}_hands.png" if plotted else "")
          + (f", {rec.record_time}_frames.png" if sheet else ""))


if __name__ == "__main__":
    main()
