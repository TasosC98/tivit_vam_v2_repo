"""Where do predicted notes match the reference? Time lag and pitch shift per recording.

A transcription that is right but displaced -- in time (video vs label timing,
frame rate) or in pitch (keyboard crop shifted by some keys) -- scores low at
lag 0 / shift 0 yet high at the right lag / shift. This slides the predicted
onsets over the reference, per recording, from saved MIDI files (CPU only, no
model), and reports per subset:

  * F1 at lag 0 (onset + pitch within ``--tol``; a fast one-to-nearest match,
    within ~1 point of mir_eval), with precision and recall;
  * the best time lag (``--max_lag_ms``) and the F1 there;
  * the best pitch shift (+-``--max_shift`` semitones = keys, default an octave)
    with its own best lag (a crop shifted by k keys AND a time offset is found
    too), and the F1 of a mirrored keyboard (A0 <-> C8). A per-key model cannot
    learn labels transposed against the picture; a global model (V2N) can;
  * the best lag in the first and in the last third of the recording: a
    difference = drift (frame rate / timestamps), equal = a constant offset;
  * the share of reference notes shorter than one video frame.

Conventions: pred onset + lag matches the reference (lag < 0 = predictions late);
pred pitch + shift matches the reference (shift > 0 = predictions too low).

  python -m pianovam_vision.offset_scan --config configs/r3.yaml --split test \
      --pred out/R3E1_r3_full_test --baseline ~/V2N_r3_test --csv results/R3E1_offsets.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import load_config
from .labels import Note, read_reference
from .metadata import filter_by_split, recordings_from_cfg
from .score_midi import load_midi_notes

KEY = 10000.0     # pitch * KEY + onset: one sorted axis, notes of different pitches never meet


def _encode(notes: List[Note], shift: int = 0, t0: float = 0.0, t1: float = np.inf,
            mirror: bool = False) -> np.ndarray:
    """mirror: A0 <-> C8 (pitch -> 129 - pitch), what a keyboard crop read the
    wrong way round gives."""
    return np.sort(np.array([((129 - n.pitch) if mirror else (n.pitch + shift)) * KEY + n.onset
                             for n in notes if t0 <= n.onset < t1], dtype=np.float64))


def _hits(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    """How many values of ``a`` have a value of ``b`` within ``tol``."""
    if len(a) == 0 or len(b) == 0:
        return 0
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(a - b[i - 1]), np.abs(b[i] - a))
    if len(b) == 1:
        d = np.abs(a - b[0])
    return int((d <= tol + 1e-9).sum())


def prf(ref: np.ndarray, est: np.ndarray, tol: float, lag: float = 0.0) -> Tuple[float, float, float]:
    """(precision, recall, F1) of onset+pitch, pred shifted by ``lag`` seconds."""
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    e = est + lag
    r = _hits(ref, e, tol) / len(ref)
    p = _hits(e, ref, tol) / len(est)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_lag(ref: np.ndarray, est: np.ndarray, tol: float, lags: np.ndarray) -> Tuple[float, float]:
    """(lag in s, F1 there). Every lag within +-tol of the true one scores about the
    same, so take the CENTRE of the near-best run of lags around the maximum
    (the maximum closest to 0 when several are equal)."""
    f1s = np.array([prf(ref, est, tol, L)[2] for L in lags])
    top = np.flatnonzero(f1s >= f1s.max() - 1e-12)
    k = int(top[np.argmin(np.abs(lags[top]))])
    lo = hi = k
    near = f1s.max() - 0.005
    while lo > 0 and f1s[lo - 1] >= near:
        lo -= 1
    while hi < len(lags) - 1 and f1s[hi + 1] >= near:
        hi += 1
    c = (lo + hi) // 2
    return float(lags[c]), float(f1s[c])


def scan(ref_notes: List[Note], est_notes: List[Note], tol: float, lags: np.ndarray,
         max_shift: int) -> Dict[str, float]:
    ref, est = _encode(ref_notes), _encode(est_notes)
    p0, r0, f0 = prf(ref, est, tol)
    lag, f_lag = best_lag(ref, est, tol, lags)                    # time only
    joint = {s: best_lag(ref, _encode(est_notes, s), tol, lags)   # time and pitch
             for s in range(-max_shift, max_shift + 1)}
    s_best = max(joint, key=lambda s: (joint[s][1], -abs(s), -abs(joint[s][0])))
    f_mirror = best_lag(ref, _encode(est_notes, mirror=True), tol, lags)[1]
    end = max([n.onset for n in ref_notes] + [0.0])
    thirds = {}
    for name, (a, b) in (("first", (0.0, end / 3)), ("last", (2 * end / 3, np.inf))):
        rf = _encode(ref_notes, 0, a, b)
        es = _encode(est_notes, s_best, a - lags.max(), b - lags.min())  # notes that may slide in
        thirds[name] = best_lag(rf, es, tol, lags)[0] if len(rf) >= 20 else np.nan
    return {"P0": p0, "R0": r0, "F1_0": f0, "lag_ms": 1000 * lag, "F1_lag": f_lag,
            "shift": s_best, "shift_lag_ms": 1000 * joint[s_best][0], "F1_shift": joint[s_best][1],
            "F1_mirror": f_mirror,
            "lag_first_ms": 1000 * thirds["first"], "lag_last_ms": 1000 * thirds["last"]}


def short_share(notes: List[Note], fps: float) -> float:
    if not notes:
        return 0.0
    return float(np.mean([(n.offset - n.onset) < 1.0 / fps for n in notes]))


def summary(rows: List[dict], system: str, group: str) -> None:
    g = [r for r in rows if f"{system}:F1_0" in r]
    if not g:
        return
    col = lambda k: np.array([r[f"{system}:{k}"] for r in g], dtype=np.float64)
    lag, f0, fl = col("lag_ms"), col("F1_0"), col("F1_lag")
    moved = (np.abs(lag) >= 100) & (fl - f0 >= 0.10)
    drift = col("lag_last_ms") - col("lag_first_ms")
    print(f"  {system:8} {group:6} {len(g):4d} recs | at lag 0: P {100 * col('P0').mean():5.1f} "
          f"R {100 * col('R0').mean():5.1f} F1 {100 * f0.mean():5.1f} | best lag: median "
          f"{np.median(lag):+6.0f} ms, F1 {100 * fl.mean():5.1f}, {moved.sum():3d} recs moved "
          f">=100 ms | drift last-first third: median {np.nanmedian(drift):+5.0f} ms | "
          f"best pitch shift: {dict(Counter(int(s) for s in col('shift')).most_common(3))}, "
          f"F1 {100 * col('F1_shift').mean():5.1f} | mirrored keyboard: F1 "
          f"{100 * col('F1_mirror').mean():5.1f}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred", required=True, help="folder of <record>.mid predictions")
    ap.add_argument("--baseline", default=None, help="second folder (e.g. V2N's MIDI)")
    ap.add_argument("--tol", type=float, default=0.05, help="onset tolerance (s)")
    ap.add_argument("--max_lag_ms", type=float, default=2000)
    ap.add_argument("--step_ms", type=float, default=20)
    ap.add_argument("--max_shift", type=int, default=12, help="pitch shifts tried (keys)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    fps = float(cfg["labels"]["fps"])
    lags = np.arange(-args.max_lag_ms, args.max_lag_ms + 1e-6, args.step_ms) / 1000.0
    systems = {"pred": Path(args.pred).expanduser()}
    if args.baseline:
        systems["baseline"] = Path(args.baseline).expanduser()
    rows = []
    for rec in filter_by_split(recordings_from_cfg(cfg), [args.split]):
        files = {k: d / f"{rec.record_time}.mid" for k, d in systems.items()}
        if not all(p.exists() for p in files.values()):
            continue
        ref = read_reference(rec, cfg)
        row = {"record": rec.record_time, "subset": rec.subset or "-", "ref_notes": len(ref),
               "short_ref": round(short_share(ref, fps), 3)}
        for k, p in files.items():
            res = scan(ref, load_midi_notes(p), args.tol, lags, args.max_shift)
            row.update({f"{k}:{m}": (round(v, 4) if isinstance(v, float) else v)
                        for m, v in res.items()})
        rows.append(row)
        print(f"{rec.record_time:34} {row['subset']:4} " + " | ".join(
            f"{k}: F1 {100 * row[f'{k}:F1_0']:5.1f} -> {100 * row[f'{k}:F1_lag']:5.1f} at "
            f"{row[f'{k}:lag_ms']:+6.0f} ms (thirds {row[f'{k}:lag_first_ms']:+.0f}/"
            f"{row[f'{k}:lag_last_ms']:+.0f}), shift {row[f'{k}:shift']:+d} -> "
            f"{100 * row[f'{k}:F1_shift']:5.1f}" for k in systems),
            flush=True)
    if not rows:
        raise SystemExit("no recording has a prediction file in every folder")

    print(f"\n=== summary (onset+pitch within {1000 * args.tol:.0f} ms; lag: pred + lag = "
          f"reference, < 0 = predictions late; shift > 0 = predictions too low) ===")
    groups = defaultdict(list)
    for r in rows:
        groups[r["subset"]].append(r)
    for sub in sorted(groups):
        for k in systems:
            summary(groups[sub], k, sub)
        short = np.mean([r["short_ref"] for r in groups[sub]])
        print(f"  {'':8} {sub:6} reference notes shorter than one frame (1/{fps:.0f} s): "
              f"{100 * short:.1f} %")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"per-recording results -> {args.csv}")


if __name__ == "__main__":
    main()
