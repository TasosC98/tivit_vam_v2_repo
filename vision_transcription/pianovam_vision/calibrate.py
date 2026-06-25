"""Calibrate the onset/frame decode thresholds on a split to maximise note F1.

The trained model over- or under-predicts at the default 0.5 thresholds. This
runs the model ONCE per recording (the expensive part), caches the probability
rolls, then cheaply sweeps thresholds to find the pair that maximises the chosen
note-level metric. Prints the best thresholds and the override to reuse.

  python -m pianovam_vision.calibrate --config configs/tiled_best.yaml \
      --checkpoint runs/tiled_best/best.pt --split valid --target full_f1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import apply_overrides, load_config
from .decode import decode_notes
from .infer import predict_rolls
from .labels import read_tsv
from .metadata import filter_by_split, load_recordings
from .metrics import note_scores
from .model import build_model
from .video import WarpedVideo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--target", default="full_f1",
                    choices=["onset_f1", "full_f1"],
                    help="metric to maximise (full_f1 = onset+offset+pitch)")
    ap.add_argument("--onset_grid", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--frame_grid", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = apply_overrides(ckpt["cfg"], args.overrides) if args.overrides \
            else ckpt["cfg"]
    else:
        cfg = load_config(args.config, args.overrides)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(cfg["data"]["root"])
    recs = filter_by_split(load_recordings(root / cfg["data"]["metadata"]),
                           [args.split])
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in recs if r.record_time not in excl]
    kb, lab = cfg["keyboard"], cfg["labels"]
    fps, min_dur = lab["fps"], cfg["decode"]["min_duration_s"]

    # 1) Predict rolls once per recording (expensive), cache with the reference.
    cache = []
    for rec in recs:
        reader = WarpedVideo(
            rec.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"]),
            rec.corners, kb["warp_width"], kb["warp_height"], kb["grayscale"],
            fps, cfg["train"].get("max_frames_per_record", 0),
            kb.get("decode_height", 0), kb.get("read_chunk", 8),
        )
        onset_p, frame_p, _ = predict_rolls(model, reader, cfg, device)
        ref = read_tsv(rec.tsv_path(root, cfg["data"]["tsv_dir"]), lab["offset_field"])
        cache.append((onset_p, frame_p, ref))
        print(f"  predicted rolls for {rec.record_time} ({len(onset_p)} frames)")

    onset_grid = [float(x) for x in args.onset_grid.split(",")]
    frame_grid = [float(x) for x in args.frame_grid.split(",")]

    # 2) Sweep thresholds on the cached rolls (cheap).
    print(f"\nsweeping {len(onset_grid)}x{len(frame_grid)} thresholds "
          f"(target={args.target})...")
    best = None
    results = []
    for ot in onset_grid:
        for ft in frame_grid:
            agg = defaultdict(list)
            for onset_p, frame_p, ref in cache:
                est = decode_notes(onset_p, frame_p, fps=fps,
                                   onset_threshold=ot, frame_threshold=ft,
                                   min_duration_s=min_dur)
                sc = note_scores(ref, est)
                for k, v in sc.items():
                    agg[k].append(v)
            mean = {k: float(np.mean(v)) for k, v in agg.items()}
            results.append((ot, ft, mean))
            if best is None or mean[args.target] > best[2][args.target]:
                best = (ot, ft, mean)

    # 3) Report.
    print(f"\n{'onset':>6} {'frame':>6} {'onset_f1':>9} {'full_f1':>9}")
    for ot, ft, m in results:
        star = "  <-- best" if (ot, ft) == (best[0], best[1]) else ""
        print(f"{ot:>6.2f} {ft:>6.2f} {m['onset_f1']:>9.4f} {m['full_f1']:>9.4f}{star}")

    ot, ft, m = best
    print("\n=== best thresholds ===")
    print(f"  onset_threshold = {ot:.2f}   frame_threshold = {ft:.2f}")
    print(f"  Note F1 (onset+pitch)        : {m['onset_f1']:.4f}")
    print(f"  Note F1 (onset+offset+pitch) : {m['full_f1']:.4f}")
    print("\nReuse these by appending to evaluate/infer:")
    print(f"  decode.onset_threshold={ot:.2f} decode.frame_threshold={ft:.2f}")


if __name__ == "__main__":
    main()
