"""Note-level evaluation on a split using mir_eval.

    python -m pianovam_vision.evaluate --config configs/default.yaml \
        --checkpoint runs/exp1/best.pt --split test [--save_midi out/test]

Whole videos are scored by default (``--max_frames N`` caps each video, e.g.
for a quick check); the training cap ``train.max_frames_per_record`` is ignored.

Cross-dataset (e.g. a PianoVAM-trained model on PianoYT): add
``--data_config configs/pianoyt.yaml`` -- the model/warp settings stay the
checkpoint's, only the dataset (``data:`` block) is swapped.

Scores are reported at onset tolerances 50 ms and 100 ms, with three offset
rules (see ``metrics.note_scores_protocols``), so every number can be put next
to the matching one in the literature. ``--csv`` writes one row per recording
(for paired significance tests between models).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import config_from_checkpoint
from .infer import transcribe
from .labels import build_target_rolls, read_reference
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import frame_prf, note_scores_protocols
from .midi_io import write_midi
from .model import build_model
from .strip_cache import open_reader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_config", default=None,
                    help="evaluate on this config's dataset instead of the checkpoint's "
                         "(cross-dataset / zero-shot)")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="frames per video to score (0 = whole video, the reportable number)")
    ap.add_argument("--save_midi", default=None, help="dir to dump predicted .mid")
    ap.add_argument("--csv", default=None, help="write per-recording scores here")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Model/keyboard settings come from the checkpoint's OWN config so they always
    # match the trained weights; CLI overrides still apply on top.
    cfg = config_from_checkpoint(ckpt, args.config, args.overrides, args.data_config)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), [args.split])
            if r.record_time not in excl]
    lab = cfg["labels"]
    fps = lab["fps"]
    print(f"evaluating {args.checkpoint} (epoch {ckpt.get('epoch', '?')}) on "
          f"{cfg['data'].get('format', 'pianovam')}:{args.split} ({len(recs)} recordings), "
          f"thresholds onset={cfg['decode']['onset_threshold']} "
          f"frame={cfg['decode']['frame_threshold']}")

    agg: Dict[str, List[float]] = defaultdict(list)
    rows = []
    skipped = 0
    n_ref_total = n_est_total = 0
    for rec in recs:
        # Some videos (esp. PianoYT YouTube downloads) fail to decode; skip them
        # instead of crashing the whole evaluation.
        try:
            reader = open_reader(cfg, rec, args.max_frames)
            est = transcribe(model, reader, cfg, device)
        except Exception as e:
            skipped += 1
            print(f"{rec.record_time}: [skipped] cannot decode video ({e})")
            continue

        n = len(reader)
        # With --max_frames the reader covers only the start of the video;
        # restrict the reference to that window or a short prediction is scored
        # against the full-length reference (meaningless F1). Without a cap t_max
        # spans the whole video, so nothing is dropped.
        t_max = n / fps
        ref = [nt for nt in read_reference(rec, cfg) if nt.onset < t_max]
        scores = note_scores_protocols(ref, est)

        # Frame-level (pitch-time grid) F1: rasterise both note sets to rolls.
        ref_roll, _, _ = build_target_rolls(ref, n, fps, 1, lab["min_note_frames"])
        est_roll, _, _ = build_target_rolls(est, n, fps, 1, lab["min_note_frames"])
        scores["frame_f1"] = frame_prf(est_roll, ref_roll, 0.5)["f1"]

        for k, v in scores.items():
            agg[k].append(v)
        n_ref_total += len(ref)
        n_est_total += len(est)
        rows.append({"record": rec.record_time, "seconds": round(t_max, 1),
                     "ref_notes": len(ref), "est_notes": len(est), **scores})
        print(f"{rec.record_time}: onset_f1@50={scores['onset_f1@50']:.3f} "
              f"@100={scores['onset_f1@100']:.3f} | onset+offset_f1="
              f"{scores['full_f1@50']:.3f} | frame_f1={scores['frame_f1']:.3f} "
              f"(ref={len(ref)} est={len(est)}, {t_max:.0f}s)")
        if args.save_midi:
            write_midi(est, Path(args.save_midi) / f"{rec.record_time}.mid")

    if not rows:
        print("no recordings evaluated")
        return

    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print("\n=== mean over recordings (higher = better, 0..1) ===")
    print(f"  {'onset tolerance / metric':52s} {'P':>6} {'R':>6} {'F1':>6}")
    for ms in (50, 100):
        for name, desc in (("onset", "onset+pitch"),
                           ("full", "onset+offset+pitch [mir_eval default offset]"),
                           ("offtol", f"onset+offset+pitch [offset within {ms} ms]")):
            print(f"  {f'{ms:>3} ms  {desc}':52s} {m[f'{name}_p@{ms}']:6.3f} "
                  f"{m[f'{name}_r@{ms}']:6.3f} {m[f'{name}_f1@{ms}']:6.3f}")
    print(f"  {'frame F1 (pitch x 1/fps grid)':52s} {'':6} {'':6} {m['frame_f1']:6.3f}")
    print(f"\n  headline: Note F1 (onset+pitch, 50 ms) = {m['onset_f1@50']:.4f} | "
          f"100 ms = {m['onset_f1@100']:.4f}")
    print(f"  evaluated {len(rows)} recording(s), {n_ref_total} reference notes, "
          f"{n_est_total} predicted"
          + (f"; skipped {skipped} (undecodable video)" if skipped else ""))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  per-recording scores -> {args.csv}")


if __name__ == "__main__":
    main()
