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
from .labels import build_target_rolls, read_reference
from .metadata import filter_by_split, recordings_from_cfg
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Build from the checkpoint's OWN config so architecture/keyboard settings
    # (arch, warp_width, ...) always match the trained weights. CLI overrides
    # still apply on top (e.g. train.max_frames_per_record=0 for full videos).
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        from .config import apply_overrides
        cfg = apply_overrides(ckpt["cfg"], args.overrides) if args.overrides \
            else ckpt["cfg"]
    else:
        cfg = load_config(args.config, args.overrides)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(cfg["data"]["root"])
    recs = filter_by_split(recordings_from_cfg(cfg), [args.split])
    kb, lab = cfg["keyboard"], cfg["labels"]
    fps = lab["fps"]

    agg: Dict[str, List[float]] = defaultdict(list)
    skipped = 0
    for rec in recs:
        # Some videos (esp. PianoYT YouTube downloads) fail to decode; skip them
        # instead of crashing the whole evaluation.
        try:
            reader = WarpedVideo(
                rec.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"]),
                rec.corners, kb["warp_width"], kb["warp_height"], kb["grayscale"],
                fps, cfg["train"].get("max_frames_per_record", 0),
                kb.get("decode_height", 0), kb.get("read_chunk", 8),
            )
            est = transcribe(model, reader, cfg, device)
        except Exception as e:
            skipped += 1
            print(f"{rec.record_time}: [skipped] cannot decode video ({e})")
            continue

        n = len(reader)
        # The reader may cap frames (train.max_frames_per_record); restrict the
        # reference to the transcribed window or a short prediction is scored
        # against the full-length reference (meaningless F1). With no cap
        # (max_frames_per_record=0) t_max spans the whole video, so nothing is dropped.
        t_max = n / fps
        ref = [nt for nt in read_reference(rec, cfg) if nt.onset < t_max]
        scores = note_scores(ref, est)

        # Frame-level (pitch-time grid) F1: rasterise both note sets to rolls.
        ref_roll, _, _ = build_target_rolls(ref, n, fps, 1, lab["min_note_frames"])
        est_roll, _, _ = build_target_rolls(est, n, fps, 1, lab["min_note_frames"])
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
    n_eval = len(agg.get("onset_f1", []))
    print(f"\n  evaluated {n_eval} recording(s)"
          + (f", skipped {skipped} (undecodable video)" if skipped else ""))


if __name__ == "__main__":
    main()
