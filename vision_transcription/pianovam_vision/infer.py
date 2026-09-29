"""Transcribe a video to MIDI with a trained model (video only).

Examples
--------
# By record_time (corners pulled from metadata_v2.json):
python -m pianovam_vision.infer --config configs/default.yaml \
    --checkpoint runs/exp1/best.pt --record_time 2024-02-14_19-10-09 \
    --output out/2024-02-14_19-10-09.mid

# Arbitrary video with explicit keyboard corners (LT RT RB LB):
python -m pianovam_vision.infer --config configs/default.yaml \
    --checkpoint runs/exp1/best.pt --video /path/clip.mp4 \
    --corners "121,355 1839,345 1839,558 120,564" --output out/clip.mid
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import config_from_checkpoint
from .decode import decode_with_cfg
from .labels import Note
from .metadata import index_by_record_time, recordings_from_cfg
from .midi_io import write_midi
from .model import build_model
from .strip_cache import open_reader
from .video import WarpedVideo


def parse_corners(s: str) -> np.ndarray:
    pts = []
    for tok in s.replace(";", " ").split():
        x, y = tok.split(",")
        pts.append([float(x), float(y)])
    arr = np.array(pts, dtype=np.float32)
    assert arr.shape == (4, 2), "Expect 4 corner points: LT RT RB LB"
    return arr


@torch.no_grad()
def predict_rolls(model, reader: WarpedVideo, cfg: Dict[str, Any], device: str):
    """Run the model over a video and return averaged probability rolls
    (onset_p, frame_p, vel_p) of shape (T, 88). Decoding/thresholding is left to
    the caller, so the same rolls can be re-thresholded cheaply (calibration)."""
    model.eval()
    n = len(reader)
    K = 88
    onset_acc = np.zeros((n, K), dtype=np.float64)
    frame_acc = np.zeros((n, K), dtype=np.float64)
    vel_acc = np.zeros((n, K), dtype=np.float64)
    count = np.zeros((n, 1), dtype=np.float64)
    has_vel = cfg["model"]["use_velocity"]

    win = cfg["infer"]["window_frames"]
    hop = cfg["infer"]["window_hop"]
    starts = list(range(0, max(1, n), hop))
    for s in starts:
        e = min(s + win, n)
        if e <= s:
            continue
        idx = np.arange(s, e)
        frames = reader.read_warped(idx)                  # (T,H,W,C)
        x = torch.from_numpy(frames).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1,T,C,H,W)
        out = model(x)
        onset_acc[s:e] += torch.sigmoid(out["onset_logits"])[0].cpu().numpy()
        frame_acc[s:e] += torch.sigmoid(out["frame_logits"])[0].cpu().numpy()
        if has_vel and "velocity" in out:
            vel_acc[s:e] += out["velocity"][0].cpu().numpy()
        count[s:e] += 1.0

    count = np.clip(count, 1.0, None)
    return onset_acc / count, frame_acc / count, (vel_acc / count) if has_vel else None


@torch.no_grad()
def transcribe(
    model, reader: WarpedVideo, cfg: Dict[str, Any], device: str
) -> List[Note]:
    onset_p, frame_p, vel_p = predict_rolls(model, reader, cfg, device)
    return decode_with_cfg(onset_p, frame_p, vel_p, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--record_time", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--corners", default=None, help='"x,y x,y x,y x,y" = LT RT RB LB')
    ap.add_argument("--data_config", default=None,
                    help="look --record_time up in this config's dataset")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="frames to transcribe (0 = whole video)")
    ap.add_argument("--output", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Build from the checkpoint's own config so arch/keyboard match the weights.
    cfg = config_from_checkpoint(ckpt, args.config, args.overrides, args.data_config)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    kb, lab = cfg["keyboard"], cfg["labels"]
    if args.record_time:
        rec = index_by_record_time(recordings_from_cfg(cfg))[args.record_time]
        reader = open_reader(cfg, rec, args.max_frames)
        source = f"{args.record_time} ({reader.source})"
    else:
        assert args.video and args.corners, "Provide --record_time OR --video + --corners"
        reader = WarpedVideo(
            Path(args.video), parse_corners(args.corners), kb["warp_width"],
            kb["warp_height"], kb["grayscale"], lab["fps"], args.max_frames,
            kb.get("decode_height", 0), kb.get("read_chunk", 8),
        )
        source = args.video
    print(f"transcribing {source} ({len(reader)} frames @ {lab['fps']} fps)")
    notes = transcribe(model, reader, cfg, device)
    write_midi(notes, args.output)
    print(f"wrote {len(notes)} notes -> {args.output}")


if __name__ == "__main__":
    main()
