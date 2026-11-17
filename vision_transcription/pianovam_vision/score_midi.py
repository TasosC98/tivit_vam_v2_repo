"""Score folders of predicted MIDI against the dataset labels (and each other).

Use it to put another system's released predictions next to ours under the
SAME metric code -- e.g. V2N ships its PianoVAM test predictions
(github.com/yonghyunk1m/V2N, predicted_midi/pianovam_test):

  python -m pianovam_vision.score_midi --config configs/vam_full.yaml --split test \
      --pred out/vam_full_test --baseline /path/to/V2N/predicted_midi/pianovam_test \
      --csv results/E1_vs_v2n.csv

Predictions are ``<record_time>.mid`` (what ``evaluate --save_midi`` writes).
Reports onset / onset+offset F1 at 50 and 100 ms (see
``metrics.note_scores_protocols``), frame F1 on a ``--frame_hz`` grid (60 Hz
= V2N's protocol) and V2N's velocity metrics (++Vel, ++Off++Vel; meaningful only
for MIDI with predicted velocities), averaged over recordings; with ``--baseline``
also a paired Wilcoxon test per metric (the significance test V2N reports).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from .config import load_config
from .labels import Note, build_target_rolls, read_reference
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import frame_prf, note_scores_protocols

METRICS = [
    ("onset_f1@50", "Onset F1 @50 ms"),
    ("onset_f1@100", "Onset F1 @100 ms"),
    ("offtol_f1@50", "Onset+offset F1 @50 ms (offset within 50 ms, V2N)"),
    ("offtol_f1@100", "Onset+offset F1 @100 ms (offset within 100 ms)"),
    ("full_f1@50", "Onset+offset F1 @50 ms (mir_eval default)"),
    ("frame_f1", "Frame F1"),
    ("vel_f1@50", "Onset+velocity F1 @50 ms (V2N ++Vel)"),
    ("vel_f1@100", "Onset+velocity F1 @100 ms"),
    ("offvel_f1@50", "Onset+offset+velocity F1 @50 ms (V2N ++Off++Vel)"),
    ("offvel_f1@100", "Onset+offset+velocity F1 @100 ms"),
    ("velmm_f1@50", "  the same, min-max velocity scaling: ++Vel @50 ms"),
    ("velmm_f1@100", "  ++Vel @100 ms (min-max)"),
    ("offvelmm_f1@50", "  ++Off++Vel @50 ms (min-max)"),
    ("offvelmm_f1@100", "  ++Off++Vel @100 ms (min-max)"),
]


def load_midi_notes(path: Path) -> List[Note]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [Note(float(n.start), float(max(n.end, n.start)), int(n.pitch), int(n.velocity))
             for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def score(ref: List[Note], est: List[Note], frame_hz: float) -> dict:
    s = note_scores_protocols(ref, est)
    t = int(np.ceil(max([n.offset for n in ref + est] + [0.0]) * frame_hz)) + 2
    r, _, _ = build_target_rolls(ref, t, frame_hz, 1, 1)
    e, _, _ = build_target_rolls(est, t, frame_hz, 1, 1)
    s["frame_f1"] = frame_prf(e, r, 0.5)["f1"]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred", required=True, help="folder of <record_time>.mid predictions")
    ap.add_argument("--baseline", default=None, help="second folder to compare against")
    ap.add_argument("--frame_hz", type=float, default=60.0,
                    help="frame-F1 grid (60 = V2N's protocol)")
    ap.add_argument("--csv", default=None, help="per-recording scores")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), [args.split])
            if r.record_time not in excl]
    systems = {"pred": Path(args.pred)}
    if args.baseline:
        systems["baseline"] = Path(args.baseline)

    per = {k: [] for k in systems}
    velocities = {k: set() for k in systems}
    rows = []
    for rec in recs:
        files = {k: d / f"{rec.record_time}.mid" for k, d in systems.items()}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            print(f"{rec.record_time}: [skipped] missing {missing}")
            continue
        ref = read_reference(rec, cfg)
        row = {"record": rec.record_time, **({"subset": rec.subset} if rec.subset else {}),
               "ref_notes": len(ref)}
        for k, p in files.items():
            notes = load_midi_notes(p)
            velocities[k].update(n.velocity for n in notes)
            s = score(ref, notes, args.frame_hz)
            per[k].append(s)
            row.update({f"{k}:{m}": round(float(s[m]), 4) for m, _ in METRICS})
        rows.append(row)
        print(f"{rec.record_time}: " + " | ".join(
            f"{k} onset@50={per[k][-1]['onset_f1@50']:.3f}" for k in systems))
    if not rows:
        raise SystemExit("nothing scored")
    for k, v in velocities.items():
        print(f"{k}: {len(v)} distinct velocities" + (f" ({min(v)}..{max(v)})" if v else "")
              + (" -- constant, so the velocity scores mean nothing" if len(v) <= 1 else ""))

    def table(idx: List[int], title: str) -> None:
        head = f"\n{title:54s} {'pred':>7}"
        if args.baseline:
            head += f" {'baseline':>9} {'Wilcoxon p':>11}"
        print(head)
        for m, desc in METRICS:
            a = np.array([per["pred"][i][m] for i in idx])
            line = f"{desc:54s} {100 * a.mean():7.1f}"
            if args.baseline:
                from scipy.stats import wilcoxon

                b = np.array([per["baseline"][i][m] for i in idx])
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                line += f" {100 * b.mean():9.1f} {p:11.3f}"
            print(line)
        if args.baseline:
            for m in ("onset_f1@50", "onset_f1@100"):
                wins = sum(per["pred"][i][m] > per["baseline"][i][m] for i in idx)
                print(f"pred beats baseline on {m} in {wins}/{len(idx)} recordings")

    table(list(range(len(rows))), f"mean over {len(rows)} recordings")
    # R3: R3s and R3x are reported separately in the literature (V2N, PPAN)
    subsets = sorted({r["subset"] for r in rows if r.get("subset")})
    if len(subsets) > 1:
        for s in subsets:
            idx = [i for i, r in enumerate(rows) if r.get("subset") == s]
            table(idx, f"subset {s}: mean over {len(idx)} recordings")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"per-recording scores -> {args.csv}")


if __name__ == "__main__":
    main()
