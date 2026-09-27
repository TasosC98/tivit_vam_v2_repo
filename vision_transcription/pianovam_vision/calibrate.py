"""Calibrate the onset/frame decode thresholds on a split to maximise note F1.

The trained model over- or under-predicts at the default 0.5 thresholds. This
runs the model ONCE per recording (the expensive part), caches the probability
rolls, then cheaply sweeps thresholds to find the pair that maximises the chosen
note-level metric. Prints the best thresholds and the override to reuse.

  python -m pianovam_vision.calibrate --config configs/tiled_best.yaml \
      --checkpoint runs/tiled_best/best.pt --split valid --target full_f1

Always calibrate on ``valid`` (never on ``test``). Whole videos are used by
default; ``--max_frames 1800`` (1 min per video) is a faster approximation.
``--data_config`` calibrates on another dataset (cross-dataset runs).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from .config import config_from_checkpoint
from .decode import decode_notes
from .infer import predict_rolls
from .labels import read_reference
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import note_scores
from .model import build_model
from .strip_cache import open_reader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--data_config", default=None,
                    help="calibrate on this config's dataset instead of the checkpoint's")
    ap.add_argument("--target", default="full_f1",
                    choices=["onset_f1", "full_f1"],
                    help="metric to maximise (full_f1 = onset+offset+pitch)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="onset tolerance (s) used while calibrating")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="frames per video (0 = whole video)")
    # Both ends matter. On unfamiliar videos (cross-dataset) the model is less
    # confident, so the best threshold can fall well below 0.2; in-domain, the
    # onset_pos_weight of 8 pushes onset probabilities up, so it can exceed 0.9.
    ap.add_argument("--onset_grid",
                    default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--frame_grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt, args.config, args.overrides, args.data_config)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    recs = filter_by_split(recordings_from_cfg(cfg), [args.split])
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in recs if r.record_time not in excl]
    fps, min_dur = cfg["labels"]["fps"], cfg["decode"]["min_duration_s"]

    # 1) Predict rolls once per recording (expensive), cache with the reference.
    cache = []
    for rec in recs:
        try:
            reader = open_reader(cfg, rec, args.max_frames)
            onset_p, frame_p, _ = predict_rolls(model, reader, cfg, device)
        except Exception as e:
            print(f"  {rec.record_time}: [skipped] cannot decode video ({e})")
            continue
        ref = read_reference(rec, cfg)
        # The rolls may cover only the first len(onset_p) frames (--max_frames).
        # Restrict the reference to that same time window, or F1 is meaningless
        # (a 20 s prediction vs a 10 min reference).
        t_max = len(onset_p) / fps
        ref = [n for n in ref if n.onset < t_max]
        cache.append((onset_p, frame_p, ref))
        print(f"  predicted rolls for {rec.record_time} "
              f"({len(onset_p)} frames, {len(ref)} ref notes in window)")
    if not cache:
        raise SystemExit("no recordings could be read")

    onset_grid = [float(x) for x in args.onset_grid.split(",")]
    frame_grid = [float(x) for x in args.frame_grid.split(",")]

    # 2) Sweep thresholds on the cached rolls (cheap).
    print(f"\nsweeping {len(onset_grid)}x{len(frame_grid)} thresholds "
          f"(target={args.target}, onset tolerance {args.tolerance * 1000:.0f} ms)...")
    best = None
    results = []
    for ot in onset_grid:
        for ft in frame_grid:
            agg = defaultdict(list)
            for onset_p, frame_p, ref in cache:
                est = decode_notes(onset_p, frame_p, fps=fps,
                                   onset_threshold=ot, frame_threshold=ft,
                                   min_duration_s=min_dur)
                sc = note_scores(ref, est, onset_tolerance=args.tolerance)
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
    if ot in (onset_grid[0], onset_grid[-1]) and len(onset_grid) > 1:
        print(f"\nNOTE: best onset threshold {ot} is at the edge of the grid; "
              f"widen --onset_grid to be sure it is the optimum.")
    print("\n=== best thresholds ===")
    print(f"  onset_threshold = {ot:.2f}   frame_threshold = {ft:.2f}")
    print(f"  Note F1 (onset+pitch)        : {m['onset_f1']:.4f}")
    print(f"  Note F1 (onset+offset+pitch) : {m['full_f1']:.4f}")
    print("\nReuse these by appending to evaluate/infer:")
    print(f"  decode.onset_threshold={ot:.2f} decode.frame_threshold={ft:.2f}")


if __name__ == "__main__":
    main()
