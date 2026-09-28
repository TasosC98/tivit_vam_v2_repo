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
    ap.add_argument("--full_grid", action="store_true",
                    help="score every onset x frame pair (slow) instead of tuning "
                         "one threshold at a time")
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
              f"({len(onset_p)} frames, {len(ref)} ref notes in window)", flush=True)
    if not cache:
        raise SystemExit("no recordings could be read")

    # High thresholds first: they give few notes (cheap to score), and in-domain
    # the optimum is usually high, which lets the bound below skip the rest.
    onset_grid = sorted({float(x) for x in args.onset_grid.split(",")}, reverse=True)
    frame_grid = sorted({float(x) for x in args.frame_grid.split(",")})
    n_ref = np.array([len(ref) for _, _, ref in cache], dtype=np.float64)
    # Each rising edge of the thresholded onset roll starts exactly one note when
    # notes of one frame survive min_duration_s, so the note count is exact.
    exact_counts = min_dur <= 1.0 / fps + 1e-9

    def note_counts(ot: float) -> np.ndarray:
        n = []
        for onset_p, _, _ in cache:
            b = onset_p >= ot
            n.append(int(b[0].sum()) + int((b[1:] & ~b[:-1]).sum()))
        return np.asarray(n, dtype=np.float64)

    results = {}           # (onset thr, frame thr) -> mean scores, or None if skipped
    best = None
    # Ties on the target (e.g. every offset score 0) are broken by the other F1.
    other = "onset_f1" if args.target == "full_f1" else "full_f1"

    def better(m, b) -> bool:
        return (m[args.target], m[other]) > (b[args.target], b[other])

    def evaluate(ot: float, ft: float) -> None:
        nonlocal best
        if (ot, ft) in results:
            return
        if exact_counts and best is not None:
            # F1 = 2*matches/(n_ref+n_est) <= 2*min(n_ref,n_est)/(n_ref+n_est): a
            # threshold predicting far too many (or too few) notes cannot win,
            # and scoring it is the slow part (mir_eval compares every pair).
            n_est = note_counts(ot)
            bound = float(np.mean(2 * np.minimum(n_ref, n_est)
                                  / np.maximum(n_ref + n_est, 1)))
            if bound < best[2][args.target]:
                results[(ot, ft)] = None
                print(f"  onset {ot:.2f} frame {ft:.2f}: skipped (cannot exceed "
                      f"{bound:.3f})", flush=True)
                return
        agg = defaultdict(list)
        for onset_p, frame_p, ref in cache:
            est = decode_notes(onset_p, frame_p, fps=fps,
                               onset_threshold=ot, frame_threshold=ft,
                               min_duration_s=min_dur)
            sc = note_scores(ref, est, onset_tolerance=args.tolerance)
            for k, v in sc.items():
                agg[k].append(v)
        mean = {k: float(np.mean(v)) for k, v in agg.items()}
        results[(ot, ft)] = mean
        print(f"  onset {ot:.2f} frame {ft:.2f}: onset_f1 {mean['onset_f1']:.4f}  "
              f"full_f1 {mean['full_f1']:.4f}", flush=True)
        if best is None or better(mean, best[2]):
            best = (ot, ft, mean)

    # 2) Search thresholds on the cached rolls. Onset F1 does not depend on the
    # frame threshold, so tune the onset threshold first, then the frame
    # threshold, then (for full_f1) the onset threshold once more.
    print(f"\nsearching thresholds (target={args.target}, onset tolerance "
          f"{args.tolerance * 1000:.0f} ms)...", flush=True)
    if args.full_grid:
        for ot in onset_grid:
            for ft in frame_grid:
                evaluate(ot, ft)
    else:
        ft0 = 0.5 if 0.5 in frame_grid else frame_grid[len(frame_grid) // 2]
        for ot in onset_grid:
            evaluate(ot, ft0)
        for ft in frame_grid:
            evaluate(best[0], ft)
        if args.target == "full_f1":
            ft1 = best[1]
            for ot in onset_grid:
                evaluate(ot, ft1)

    # 3) Report.
    print(f"\n{'onset':>6} {'frame':>6} {'onset_f1':>9} {'full_f1':>9}")
    for (ot, ft), m in sorted(results.items()):
        if m is None:
            print(f"{ot:>6.2f} {ft:>6.2f} {'skipped':>9} {'':>9}")
            continue
        star = "  <-- best" if (ot, ft) == (best[0], best[1]) else ""
        print(f"{ot:>6.2f} {ft:>6.2f} {m['onset_f1']:>9.4f} {m['full_f1']:>9.4f}{star}")

    ot, ft, m = best
    if ot in (min(onset_grid), max(onset_grid)) and len(onset_grid) > 1:
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
