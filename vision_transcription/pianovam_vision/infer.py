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

from .config import load_config
from .decode import decode_notes
from .labels import Note
from .metadata import index_by_record_time, load_recordings
from .midi_io import write_midi
from .model import build_model
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
def transcribe(
    model, reader: WarpedVideo, cfg: Dict[str, Any], device: str
) -> List[Note]:
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
    onset_p = onset_acc / count
    frame_p = frame_acc / count
    vel_p = (vel_acc / count) if has_vel else None

    d = cfg["decode"]
    return decode_notes(
        onset_p, frame_p, fps=cfg["labels"]["fps"],
        onset_threshold=d["onset_threshold"], frame_threshold=d["frame_threshold"],
        min_duration_s=d["min_duration_s"], velocity_probs=vel_p,
        default_velocity=d["default_velocity"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--record_time", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--corners", default=None, help='"x,y x,y x,y x,y" = LT RT RB LB')
    ap.add_argument("--output", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    root = Path(cfg["data"]["root"])
    if args.record_time:
        recs = index_by_record_time(load_recordings(root / cfg["data"]["metadata"]))
        rec = recs[args.record_time]
        video_path = rec.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        corners = rec.corners
    else:
        assert args.video and args.corners, "Provide --record_time OR --video + --corners"
        video_path = Path(args.video)
        corners = parse_corners(args.corners)

    kb, lab = cfg["keyboard"], cfg["labels"]
    reader = WarpedVideo(
        video_path, corners, kb["warp_width"], kb["warp_height"],
        kb["grayscale"], lab["fps"], cfg["train"].get("max_frames_per_record", 0),
    )
    print(f"transcribing {video_path} ({len(reader)} frames @ {lab['fps']} fps)")
    notes = transcribe(model, reader, cfg, device)
    write_midi(notes, args.output)
    print(f"wrote {len(notes)} notes -> {args.output}")


if __name__ == "__main__":
    main()
