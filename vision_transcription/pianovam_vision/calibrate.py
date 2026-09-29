"""Calibrate the decode settings on a split to maximise note F1.

The trained model over- or under-predicts at the default 0.5 thresholds. This
runs the model ONCE per recording (the expensive part), caches the probability
rolls, then cheaply sweeps thresholds to find the pair that maximises the chosen
note-level metric. Prints the best settings and the override line to reuse.

  python -m pianovam_vision.calibrate --config configs/tiled_best.yaml \
      --checkpoint runs/tiled_best/best.pt --split valid --target full_f1

Always calibrate on ``valid`` (never on ``test``). Whole videos are used by
default; ``--max_frames 1800`` (1 min per video) is a faster approximation.
``--data_config`` calibrates on another dataset (cross-dataset runs).

``--timing`` also tunes WHEN notes start, on the calibrated thresholds: onset
timing "first" (first frame above the threshold) or "centroid" (sub-frame centre
of the onset peak), one time shift for all keys, then an extra shift for black
keys (see ``decode.decode_notes``). ``--shift_grid`` sets the shifts tried
(seconds, "start:stop:step"); widen it for cross-dataset runs, whose labels can
sit ~0.1 s away from what the model sees. ``--rolls_dir`` stores the model's
outputs so a later run decodes without the GPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from .config import config_from_checkpoint
from .decode import decode_notes, parse_grid
from .infer import predict_rolls
from .labels import read_reference
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import note_scores
from .model import build_model
from .rolls import RollsCache, rolls_meta
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
    ap.add_argument("--timing", action="store_true",
                    help="also tune onset timing and time shifts (see module docstring)")
    ap.add_argument("--shift_grid", default="-0.04:0.04:0.01",
                    help="time shifts (s) for all keys, 'start:stop:step' or 'a,b,c'")
    ap.add_argument("--black_grid", default="-0.03:0.03:0.01",
                    help="extra time shifts (s) for black keys")
    ap.add_argument("--rolls_dir", default=None,
                    help="store/reuse the model's probability rolls here")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt, args.config, args.overrides, args.data_config)

    model = None

    def get_model():
        nonlocal model
        if model is None:
            model = build_model(cfg).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
        return model

    recs = filter_by_split(recordings_from_cfg(cfg), [args.split])
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in recs if r.record_time not in excl]
    fps, min_dur = cfg["labels"]["fps"], cfg["decode"]["min_duration_s"]
    rolls = RollsCache(args.rolls_dir, rolls_meta(args.checkpoint, cfg, args.split, args.max_frames)) \
        if args.rolls_dir else None

    # 1) Predict rolls once per recording (expensive), cache with the reference.
    cache = []
    for rec in recs:
        try:
            got = rolls.load(rec.record_time) if rolls else None
            if got is None:
                reader = open_reader(cfg, rec, args.max_frames)
                onset_p, frame_p, _ = predict_rolls(get_model(), reader, cfg, device)
                if rolls:
                    rolls.save(rec.record_time, onset_p, frame_p)
                how = "predicted"
            else:
                onset_p, frame_p = got
                how = "loaded"
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
        print(f"  {how} rolls for {rec.record_time} "
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

    def mean_scores(ot: float, ft: float, timing=("first", 0.0, 0.0)) -> dict:
        agg = defaultdict(list)
        for onset_p, frame_p, ref in cache:
            est = decode_notes(onset_p, frame_p, fps=fps,
                               onset_threshold=ot, frame_threshold=ft,
                               min_duration_s=min_dur, onset_timing=timing[0],
                               time_shift_s=timing[1], black_shift_s=timing[2])
            sc = note_scores(ref, est, onset_tolerance=args.tolerance)
            for k, v in sc.items():
                agg[k].append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}

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
        mean = mean_scores(ot, ft)
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

    # 4) Timing, on the chosen thresholds. The note count does not depend on the
    # timing, so there is nothing to prune; one shift for every key first, then
    # an extra shift for black keys at that shift, for each onset timing rule.
    timing = ("first", 0.0, 0.0)
    if args.timing:
        shifts, blacks = parse_grid(args.shift_grid), parse_grid(args.black_grid)
        print(f"\nsearching onset timing at onset {ot:.2f} frame {ft:.2f} "
              f"({len(shifts)} shifts, {len(blacks)} black-key shifts)...", flush=True)
        tried = {timing: m}
        for mode in ("first", "centroid"):
            mode_best = None
            for s in shifts:
                key = (mode, s, 0.0)
                if key not in tried:
                    tried[key] = mean_scores(ot, ft, key)
                    print(f"  {mode:8s} shift {s * 1000:+5.0f} ms: onset_f1 "
                          f"{tried[key]['onset_f1']:.4f}  full_f1 {tried[key]['full_f1']:.4f}",
                          flush=True)
                if mode_best is None or better(tried[key], tried[mode_best]):
                    mode_best = key
            for b in blacks:
                key = (mode, mode_best[1], b)
                if key not in tried:
                    tried[key] = mean_scores(ot, ft, key)
                    print(f"  {mode:8s} shift {mode_best[1] * 1000:+5.0f} ms, black keys "
                          f"{b * 1000:+4.0f} ms: onset_f1 {tried[key]['onset_f1']:.4f}  "
                          f"full_f1 {tried[key]['full_f1']:.4f}", flush=True)
        for key, sc in tried.items():
            if better(sc, tried[timing]):
                timing = key
        m = tried[timing]
        print(f"\nbest timing: {timing[0]}, shift {timing[1] * 1000:+.0f} ms, black keys "
              f"{timing[2] * 1000:+.0f} ms more "
              f"(onset_f1 {tried[('first', 0.0, 0.0)]['onset_f1']:.4f} -> {m['onset_f1']:.4f}, "
              f"full_f1 {tried[('first', 0.0, 0.0)]['full_f1']:.4f} -> {m['full_f1']:.4f})")
        if timing[1] in (min(shifts), max(shifts)) and len(shifts) > 1:
            print(f"NOTE: best shift {timing[1]} s is at the edge of --shift_grid; widen it.")
        if timing[2] in (min(blacks), max(blacks)) and len(blacks) > 1 and timing[2] != 0.0:
            print(f"NOTE: best black-key shift {timing[2]} s is at the edge of --black_grid.")

    print("\n=== best settings ===")
    print(f"  onset_threshold = {ot:.2f}   frame_threshold = {ft:.2f}")
    if args.timing:
        print(f"  onset_timing = {timing[0]}   time_shift_s = {timing[1]:+.3f}   "
              f"black_shift_s = {timing[2]:+.3f}")
    print(f"  Note F1 (onset+pitch)        : {m['onset_f1']:.4f}")
    print(f"  Note F1 (onset+offset+pitch) : {m['full_f1']:.4f}")
    print("\nReuse these by appending to evaluate/infer:")
    line = f"decode.onset_threshold={ot:.2f} decode.frame_threshold={ft:.2f}"
    if args.timing:
        line += (f" decode.onset_timing={timing[0]} decode.time_shift_s={timing[1]:.3f}"
                 f" decode.black_shift_s={timing[2]:.3f}")
    print(f"  {line}")


if __name__ == "__main__":
    main()
