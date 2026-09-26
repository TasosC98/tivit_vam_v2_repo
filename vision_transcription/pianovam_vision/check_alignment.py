"""Check that each video's keyboard crop lines up with the 88-key grid.

The per-key model reads key k from column k of the warped strip, so the strip
must start exactly at A0 and end at C8. This measures it for every video: the
median of frames sampled across the video (hands move, keys do not) gives a
clean keyboard image; in each horizontal band the dark-column profile is matched
against the real black-key pattern of an 88-key piano (``keyboard.key_geometry``)
with a horizontal scale and shift. The report gives the error at the A0 and C8
ends in white keys; upside-down strips and crops where no keyboard is found are
flagged. Black keys stand ~1 cm above the white keys, so a camera above the
piano sees them slightly spread out (parallax): even a perfect crop reads a few
tenths of a key off at the ends. Hence errors are flagged only beyond one white
key, and the PianoVAM crops (hand-annotated) are the reference for "aligned" --
run it on both datasets. (A mirrored keyboard cannot be told apart from a shift
of 3-4 white keys -- the 2-3 black-key grouping repeats every octave -- so it
shows up as a large misalignment.)

  python -m pianovam_vision.check_alignment --config configs/yt_full.yaml \
      --csv results/alignment_yt.csv

Uses the strip cache when built (else decodes the sampled frames). The CSV also
holds suggested corrected keyboard corners (x-extent only) for later use.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import load_config
from .keyboard import key_geometry
from .metadata import filter_by_split, recordings_from_cfg
from .strip_cache import open_reader

N_BANDS = 10             # horizontal bands searched for the black keys
MISALIGNED_KEYS = 1.0    # error above this many white keys at either end -> flagged
MIN_FIT = 0.3            # correlation below this -> black-key pattern not found


def black_key_template(width: int) -> np.ndarray:
    """1 where a black key is, 0 elsewhere, along a canonical 88-key strip."""
    t = np.zeros(width, dtype=np.float64)
    for kind, x0, _, x1, _ in key_geometry(width, 100).values():
        if kind == "black":
            t[max(0, x0):min(width, x1)] = 1.0
    return t


def darkness(profile: np.ndarray) -> np.ndarray:
    """Standardised darkness along the strip, slow lighting trends removed."""
    from scipy.ndimage import median_filter

    d = -(profile - median_filter(profile, size=81, mode="nearest"))
    return (d - d.mean()) / (d.std() + 1e-9)


class LayoutFitter:
    """Best horizontal scale/shift of the black-key template for a profile.

    Observed column x shows canonical column (x - c - shift) / scale + c.
    """

    def __init__(self, width: int, scales=np.arange(0.90, 1.10001, 0.0025),
                 max_shift_keys: float = 3.5):
        t = black_key_template(width)
        self.w = width
        self.max_shift = int(round(max_shift_keys * width / 52))
        c = width / 2.0
        xe = np.arange(-self.max_shift, width + self.max_shift)
        ones = np.ones(width)
        self.scales, self.ts, self.norms = list(scales), [], []
        for s in self.scales:
            u = np.round((xe - c) / s + c).astype(np.int64)
            ts = np.where((u >= 0) & (u < width), t[np.clip(u, 0, width - 1)], 0.0)
            n1 = np.correlate(ts, ones, mode="valid")          # black pixels per shift
            self.ts.append(ts)
            self.norms.append(np.sqrt(np.maximum(n1 - n1 ** 2 / width, 1e-9)))

    def fit(self, d: np.ndarray) -> Tuple[float, float, int]:
        """(correlation, scale, shift_px) of the best match."""
        dn = np.linalg.norm(d) + 1e-9
        best = (-np.inf, 1.0, 0)
        for s, ts, norm in zip(self.scales, self.ts, self.norms):
            r = np.correlate(ts, d, mode="valid") / (dn * norm)   # index k <-> shift max_shift-k
            k = int(np.argmax(r))
            if r[k] > best[0]:
                best = (float(r[k]), float(s), int(self.max_shift - k))
        return best


_FITTERS: Dict[int, LayoutFitter] = {}


def analyse(strips: np.ndarray) -> Dict:
    """Alignment of a keyboard crop from (K, H, W, C) uint8 warped strips."""
    gray = strips[..., :3].astype(np.float64).mean(axis=3)
    bg = np.median(gray, axis=0)                         # hands removed
    h, w = bg.shape
    profiles = []
    for b in range(N_BANDS):
        lo = int(b * h / N_BANDS)
        profiles.append(darkness(bg[lo:max(int((b + 1) * h / N_BANDS), lo + 1)].mean(axis=0)))
    fitter = _FITTERS.setdefault(w, LayoutFitter(w))
    best = None
    for d in profiles:
        r, s, sh = fitter.fit(d)
        if best is None or r > best["fit"]:
            best = {"fit": r, "scale": s, "shift_px": sh}
    # Which bands show the black keys at that fit? They sit at the back of the
    # keyboard, i.e. at the top of a correctly oriented strip.
    k = fitter.max_shift - best["shift_px"]
    tw = fitter.ts[fitter.scales.index(best["scale"])][k:k + w]
    rs = [np.nan_to_num(np.corrcoef(d, tw)[0, 1]) if tw.std() > 0 else 0.0 for d in profiles]
    zone = [b for b, r in enumerate(rs) if r > 0.5 * best["fit"]] or [0]
    best["band_center"] = (float(np.mean(zone)) + 0.5) / N_BANDS
    wk = w / 52.0                                         # white-key width in px
    c = w / 2.0
    x0 = -c * best["scale"] + c + best["shift_px"]        # where A0's left edge appears
    x1 = c * best["scale"] + c + best["shift_px"]         # where C8's right edge appears
    at_limit = (best["scale"] in (fitter.scales[0], fitter.scales[-1])
                or abs(best["shift_px"]) >= fitter.max_shift)
    best.update(x0=x0, x1=x1, err_left_keys=x0 / wk, err_right_keys=(x1 - w) / wk,
                at_limit=at_limit)
    best["verdict"] = verdict(best)
    return best


def verdict(res: Dict) -> str:
    if res["fit"] < MIN_FIT:
        return "NO_KEYS_FOUND"
    if res["band_center"] > 0.6:
        return "UPSIDE_DOWN?"
    if max(abs(res["err_left_keys"]), abs(res["err_right_keys"])) > MISALIGNED_KEYS:
        return "MISALIGNED"
    return "OK"


def corrected_corners(corners: np.ndarray, x0: float, x1: float, w: int) -> np.ndarray:
    """Move the left/right keyboard edges to where A0 and C8 were found."""
    lt, rt, rb, lb = [np.asarray(p, dtype=np.float64) for p in corners]
    a, b = x0 / w, x1 / w
    return np.array([lt + (rt - lt) * a, lt + (rt - lt) * b,
                     lb + (rb - lb) * b, lb + (rb - lb) * a])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="default: the config's train+valid+test splits")
    ap.add_argument("--frames", type=int, default=24, help="frames sampled per video")
    ap.add_argument("--csv", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    splits = args.splits or (list(d["train_splits"]) + list(d["valid_splits"])
                             + list(d["test_splits"]))
    excl = set(d.get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), splits)
            if r.record_time not in excl]

    rows: List[Dict] = []
    for rec in recs:
        try:
            reader = open_reader(cfg, rec, 0)
            n = len(reader)
            idx = np.linspace(0.05 * (n - 1), 0.95 * (n - 1), args.frames).round().astype(int)
            res = analyse(reader.read_warped(idx))
        except Exception as e:
            print(f"{rec.record_time:>20}: [skipped] {e}", flush=True)
            continue
        w = cfg["keyboard"]["warp_width"]
        fixed = corrected_corners(rec.corners, res["x0"], res["x1"], w)
        rows.append({
            "record": rec.record_time, "split": rec.split, "verdict": res["verdict"],
            "fit": round(res["fit"], 3),
            "err_A0_end_keys": round(res["err_left_keys"], 2),
            "err_C8_end_keys": round(res["err_right_keys"], 2),
            "scale": round(res["scale"], 4), "shift_px": res["shift_px"],
            "black_keys_band_pct": round(100 * res["band_center"]),
            "at_search_limit": res["at_limit"],
            "fixed_corners": " ".join(f"{x:.1f},{y:.1f}" for x, y in fixed),
        })
        r = rows[-1]
        print(f"{rec.record_time:>20} [{rec.split:>5}] {r['verdict']:<13} fit {r['fit']:.2f} | "
              f"A0 end {r['err_A0_end_keys']:+.2f}  C8 end {r['err_C8_end_keys']:+.2f} "
              f"white keys" + ("  (at search limit)" if r["at_search_limit"] else ""),
              flush=True)
    if not rows:
        raise SystemExit("nothing analysed")

    counts = Counter(r["verdict"] for r in rows)
    ok = [r for r in rows if r["verdict"] == "OK"]
    worst = [max(abs(r["err_A0_end_keys"]), abs(r["err_C8_end_keys"])) for r in ok]
    print(f"\n{len(rows)} videos: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    if worst:
        print(f"aligned videos: median worst-end error {np.median(worst):.2f} white keys "
              f"(a model column is {52 / 88:.2f} white keys wide)")
    for k in ("NO_KEYS_FOUND", "UPSIDE_DOWN?", "MISALIGNED"):
        bad = [r["record"] for r in rows if r["verdict"] == k]
        if bad:
            print(f"  {k}: {' '.join(bad)}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"per-video results -> {args.csv}")


if __name__ == "__main__":
    main()
