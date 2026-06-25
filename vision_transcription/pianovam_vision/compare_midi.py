"""Compare predicted MIDI files against the dataset's reference MIDI and report
the percentage of notes in common (same pitch, onset within a tolerance).

Notes are matched on PITCH + ONSET only (not offset): the dataset MIDI offsets
include sustain-pedal tails that a camera cannot see, so onset+pitch is the fair
comparison for a visual model.

  python -m pianovam_vision.compare_midi --config configs/default.yaml \
      --pred_dir out/tiled_sharp [--ref_dir /path/to/MIDI] [--tol 0.05]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import load_config


def load_notes(path: str):
    """Return [(pitch, onset, offset), ...] sorted by onset, from a MIDI file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    notes = [(n.pitch, n.start, n.end)
             for inst in pm.instruments if not inst.is_drum
             for n in inst.notes]
    notes.sort(key=lambda x: (x[1], x[0]))
    return notes


def count_matches(ref, est, tol: float) -> int:
    """Greedy one-to-one match: a ref note matches an unused est note of the
    same pitch whose onset is within `tol` seconds (nearest wins)."""
    by_pitch = defaultdict(list)
    for j, (p, s, _) in enumerate(est):
        by_pitch[p].append(j)
    used = [False] * len(est)
    matched = 0
    for p, s, _ in ref:
        best, best_d = -1, tol
        for j in by_pitch.get(p, []):
            if used[j]:
                continue
            d = abs(est[j][1] - s)
            if d <= best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            matched += 1
    return matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred_dir", required=True, help="dir of predicted .mid")
    ap.add_argument("--ref_dir", default=None,
                    help="dir of reference .mid (default: dataset MIDI dir)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="onset match tolerance in seconds")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
    ref_dir = Path(args.ref_dir) if args.ref_dir else root / cfg["data"]["midi_dir"]
    pred_dir = Path(args.pred_dir)

    preds = sorted(pred_dir.glob("*.mid"))
    if not preds:
        raise SystemExit(f"no .mid files in {pred_dir}")

    tot_ref = tot_est = tot_match = 0
    print(f"matching on pitch + onset (tol={args.tol*1000:.0f} ms)\n")
    print(f"{'recording':<22} {'ref':>6} {'pred':>6} {'common':>7} "
          f"{'recall%':>8} {'prec%':>7} {'F1':>6}")
    for pf in preds:
        rf = ref_dir / pf.name
        if not rf.exists():
            print(f"{pf.stem:<22} (no reference MIDI in {ref_dir})")
            continue
        ref = load_notes(str(rf))
        est = load_notes(str(pf))
        m = count_matches(ref, est, args.tol)
        rec = m / len(ref) if ref else 0.0
        prec = m / len(est) if est else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        tot_ref += len(ref); tot_est += len(est); tot_match += m
        print(f"{pf.stem:<22} {len(ref):>6} {len(est):>6} {m:>7} "
              f"{rec*100:>7.1f}% {prec*100:>6.1f}% {f1:>6.3f}")

    if tot_ref and tot_est:
        rec = tot_match / tot_ref
        prec = tot_match / tot_est
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        print("\n=== overall (pooled over all files) ===")
        print(f"  reference notes : {tot_ref}")
        print(f"  predicted notes : {tot_est}")
        print(f"  notes in common : {tot_match}")
        print(f"  % of reference notes found (recall)   : {rec*100:.1f}%")
        print(f"  % of predicted notes correct (precision): {prec*100:.1f}%")
        print(f"  F1                                      : {f1:.3f}")


if __name__ == "__main__":
    main()
