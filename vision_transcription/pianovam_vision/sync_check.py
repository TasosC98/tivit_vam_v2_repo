"""Estimate the label <-> video time offset of each recording with a trained model.

PianoYT labels were transcribed from each video's AUDIO, so a video whose sound
and picture are not in sync has labels shifted against the frames. With a
trained model as a probe: transcribe the video, then slide the reference notes
over a range of lags and find the lag that maximises onset F1. A recording
whose best lag is far from 0 (with a clear F1 gain) is out of sync.

  python -m pianovam_vision.sync_check --config configs/pianoyt.yaml \
      --checkpoint runs/yt_full/best.pt --splits train valid test \
      --csv sync_pianoyt.csv --write_offsets pianoyt_label_offsets.json \
      decode.onset_threshold=0.6

Then set ``data.label_offsets: pianoyt_label_offsets.json`` so training,
evaluation and the preview tools all use the corrected labels.

Every probe model has its own small timing bias (it shows up as the same lag on
all in-sync videos), so corrections are made relative to the MEDIAN lag: only
recordings that deviate from the typical timing by ``--min_lag_ms`` (with an F1
gain of ``--min_gain``) are corrected, by (their lag - median). The median
itself is printed -- with a probe trained on cleanly synced data (PianoVAM) it
estimates a dataset-wide label offset. Decode with the calibrated thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import config_from_checkpoint
from .decode import decode_notes
from .infer import predict_rolls
from .labels import read_reference, shift_notes
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import _to_intervals, note_scores
from .model import build_model
from .strip_cache import open_reader


def lag_curve(ref, est, lags_s, tol: float) -> np.ndarray:
    """Onset F1 of ``est`` against ``ref`` shifted by each lag (seconds)."""
    return np.array([note_scores(shift_notes(ref, float(d)), est,
                                 onset_tolerance=tol)["onset_f1"] for d in lags_s])


def estimate_lag(ref, est, lags_s, tol: float):
    """Best label shift (s) and the F1 curve it came from.

    F1 is flat within +/- tol of the true lag, so the curve's argmax only
    locates the plateau (at its edge). Refine with the median onset error of the
    notes matched there, which lands in the middle.
    """
    import mir_eval

    curve = lag_curve(ref, est, lags_s, tol)
    lag = float(lags_s[int(np.argmax(curve))])
    est_i, est_f = _to_intervals(est)
    for _ in range(2):
        shifted = shift_notes(ref, lag)
        ref_i, ref_f = _to_intervals(shifted)
        pairs = mir_eval.transcription.match_notes(
            ref_i, ref_f, est_i, est_f, onset_tolerance=tol, offset_ratio=None)
        if not pairs:
            break
        lag += float(np.median([est_i[j, 0] - ref_i[i, 0] for i, j in pairs]))
    return lag, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--data_config", default=None,
                    help="probe this config's dataset instead of the checkpoint's")
    ap.add_argument("--max_frames", type=int, default=3600,
                    help="frames per video to probe (3600 = 2 min at 30 fps; 0 = all)")
    ap.add_argument("--max_lag_ms", type=float, default=400)
    ap.add_argument("--step_ms", type=float, default=10)
    ap.add_argument("--tolerance", type=float, default=0.05, help="onset tolerance (s)")
    ap.add_argument("--csv", default=None, help="per-recording results")
    ap.add_argument("--write_offsets", default=None,
                    help="JSON of per-recording corrections for data.label_offsets")
    ap.add_argument("--min_gain", type=float, default=0.05,
                    help="only correct a recording if F1 improves at least this much")
    ap.add_argument("--min_lag_ms", type=float, default=40,
                    help="...and its best lag is at least this far from the median lag")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt, args.config, args.overrides, args.data_config)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fps, d = cfg["labels"]["fps"], cfg["decode"]
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), args.splits)
            if r.record_time not in excl]
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + 1e-6, args.step_ms) / 1000.0
    zero = int(np.argmin(np.abs(lags)))

    rows, curves = [], []
    for rec in recs:
        try:
            reader = open_reader(cfg, rec, args.max_frames)
            onset_p, frame_p, _ = predict_rolls(model, reader, cfg, device)
        except Exception as e:
            print(f"{rec.record_time}: [skipped] {e}")
            continue
        t_max = len(onset_p) / fps
        ref = [n for n in read_reference(rec, cfg) if n.onset < t_max]
        est = decode_notes(onset_p, frame_p, fps=fps, onset_threshold=d["onset_threshold"],
                           frame_threshold=d["frame_threshold"],
                           min_duration_s=d["min_duration_s"])
        if not ref or not est:
            print(f"{rec.record_time}: [skipped] ref={len(ref)} est={len(est)} notes")
            continue
        lag, f1 = estimate_lag(ref, est, lags, args.tolerance)
        f1_best = note_scores(shift_notes(ref, lag), est,
                              onset_tolerance=args.tolerance)["onset_f1"]
        curves.append(f1)
        row = {"record": rec.record_time, "split": rec.split, "ref_notes": len(ref),
               "est_notes": len(est), "f1_at_0": round(float(f1[zero]), 4),
               "best_lag_ms": int(round(lag * 1000)),
               "f1_at_best": round(float(f1_best), 4),
               "gain": round(float(f1_best - f1[zero]), 4)}
        rows.append(row)
        print(f"{rec.record_time:>12} [{rec.split}] F1 at 0 ms = {row['f1_at_0']:.3f}, "
              f"best {row['f1_at_best']:.3f} at {row['best_lag_ms']:+d} ms", flush=True)

    if not rows:
        raise SystemExit("nothing probed")

    mean_curve = np.mean(curves, axis=0)
    g = int(np.argmax(mean_curve))
    median = float(np.median([r["best_lag_ms"] for r in rows]))
    for r in rows:
        r["lag_vs_median_ms"] = int(round(r["best_lag_ms"] - median))
        r["out_of_sync"] = (r["gain"] >= args.min_gain
                            and abs(r["lag_vs_median_ms"]) >= args.min_lag_ms)
    dev = np.array([r["lag_vs_median_ms"] for r in rows])
    print(f"\n{len(rows)} recordings probed. Median best lag {median:+.0f} ms "
          f"(probe timing bias, or a dataset-wide label offset if the probe was "
          f"trained on cleanly synced data); dataset-level curve peaks at "
          f"{lags[g] * 1000:+.0f} ms.")
    for lo, hi in ((0, 20), (20, 60), (60, 100), (100, 200), (200, 1e9)):
        k = int(((np.abs(dev) >= lo) & (np.abs(dev) < hi)).sum())
        print(f"  |lag - median| in [{lo:.0f}, {hi:.0f}) ms: {k}")
    bad = [r for r in rows if r["out_of_sync"]]
    print(f"{len(bad)} recording(s) out of sync (|lag - median| >= {args.min_lag_ms:.0f} ms "
          f"and F1 gain >= {args.min_gain}):")
    for r in bad:
        print(f"  {r['record']:>12} [{r['split']}] {r['lag_vs_median_ms']:+d} ms "
              f"(F1 {r['f1_at_0']:.3f} -> {r['f1_at_best']:.3f})")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"per-recording results -> {args.csv}")
    if args.write_offsets:
        fix = {r["record"]: r["lag_vs_median_ms"] / 1000.0 for r in bad}
        with open(args.write_offsets, "w", encoding="utf-8") as f:
            json.dump(fix, f, indent=1, sort_keys=True)
        print(f"{len(fix)} correction(s) -> {args.write_offsets} "
              f"(use with data.label_offsets={args.write_offsets})")


if __name__ == "__main__":
    main()
