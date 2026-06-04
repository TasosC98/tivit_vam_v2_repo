"""Note-level evaluation on a split using mir_eval.

    python -m pianovam_vision.evaluate --config configs/default.yaml \
        --checkpoint runs/exp1/best.pt --split test [--save_midi out/test]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import load_config
from .infer import transcribe
from .labels import build_target_rolls, read_tsv
from .metadata import filter_by_split, load_recordings
from .metrics import frame_prf, note_scores
from .midi_io import write_midi
from .model import build_model
from .video import WarpedVideo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--save_midi", default=None, help="dir to dump predicted .mid")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    root = Path(cfg["data"]["root"])
    recs = filter_by_split(load_recordings(root / cfg["data"]["metadata"]), [args.split])
    kb, lab = cfg["keyboard"], cfg["labels"]

    agg: Dict[str, List[float]] = defaultdict(list)
    for rec in recs:
        reader = WarpedVideo(
            rec.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"]),
            rec.corners, kb["warp_width"], kb["warp_height"], kb["grayscale"],
            lab["fps"], cfg["train"].get("max_frames_per_record", 0),
        )
        est = transcribe(model, reader, cfg, device)
        ref = read_tsv(rec.tsv_path(root, cfg["data"]["tsv_dir"]), lab["offset_field"])
        scores = note_scores(ref, est)

        # Frame-level (pitch-time grid) F1: rasterise both note sets to rolls.
        n = len(reader)
        ref_roll, _, _ = build_target_rolls(ref, n, lab["fps"], 1, lab["min_note_frames"])
        est_roll, _, _ = build_target_rolls(est, n, lab["fps"], 1, lab["min_note_frames"])
        scores["frame_f1"] = frame_prf(est_roll, ref_roll, 0.5)["f1"]

        for k, v in scores.items():
            agg[k].append(v)
        print(f"{rec.record_time}: note(onset)_f1={scores['onset_f1']:.3f} "
              f"note(onset+offset)_f1={scores['full_f1']:.3f} "
              f"frame_f1={scores['frame_f1']:.3f} (ref={len(ref)} est={len(est)})")
        if args.save_midi:
            write_midi(est, Path(args.save_midi) / f"{rec.record_time}.mid")

    print("\n=== mean over split (higher = better, range 0..1) ===")
    labels = [
        ("onset_p", "Note precision (onset+pitch)"),
        ("onset_r", "Note recall    (onset+pitch)"),
        ("onset_f1", "Note F1        (onset+pitch)        <- onset & pitch accuracy"),
        ("full_p", "Note precision (onset+offset+pitch)"),
        ("full_r", "Note recall    (onset+offset+pitch)"),
        ("full_f1", "Note F1        (onset+offset+pitch) <- OVERALL note score"),
        ("frame_f1", "Frame F1       (pitch-time grid)    <- per-frame pitch accuracy"),
    ]
    for k, desc in labels:
        vals = agg.get(k, [])
        if vals:
            print(f"  {desc:38s}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
