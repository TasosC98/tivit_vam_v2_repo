"""Derive which hand (Left/Right) played each note, from the MediaPipe
Handskeleton data.

The skeleton JSON maps frame_index (str, at the native video fps) ->
    {"Left": landmarks | null, "Right": landmarks | null}
where landmarks is 21 [x, y, z] points in normalized [0,1] image coordinates.

We warp each hand's fingertips through the SAME keyboard perspective transform
used for the strip, then assign each note to the hand whose fingertip is closest
to the pressed key's column at the note onset. MediaPipe's own Left/Right tag is
the handedness; we only decide *which* detected hand pressed the key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .keyboard import perspective_matrix, pitch_to_x

# MediaPipe Hands fingertip landmark indices: thumb, index, middle, ring, pinky.
FINGERTIPS = [4, 8, 12, 16, 20]


def load_skeleton(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def skeleton_path(root, record_time, handskeleton_dir: str = "Handskeleton") -> Path:
    return Path(root) / handskeleton_dir / f"{record_time}.json"


def _hand_points_strip(kpts, matrix, native_w: int, native_h: int,
                       which=FINGERTIPS) -> np.ndarray:
    """Warp selected normalized landmarks to strip pixel coords -> (N,2)."""
    import cv2
    pts = np.array(kpts, dtype=np.float32)[which, :2]
    pts[:, 0] *= native_w
    pts[:, 1] *= native_h
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def hands_at_frame(skeleton, frame_idx: int, corners, warp_w, warp_h,
                   native_w=1920, native_h=1080) -> Dict[str, np.ndarray]:
    """Fingertip positions (strip coords) for each detected hand at a frame."""
    matrix = perspective_matrix(corners, warp_w, warp_h)
    frame = skeleton.get(str(frame_idx)) or {}
    out: Dict[str, np.ndarray] = {}
    for hand in ("Left", "Right"):
        kpts = frame.get(hand)
        if kpts:
            out[hand] = _hand_points_strip(kpts, matrix, native_w, native_h)
    return out


def assign_hands(notes, skeleton, corners, warp_w, warp_h,
                 native_w=1920, native_h=1080, native_fps=60.0) -> List[str]:
    """Return 'Left' / 'Right' / 'unknown' for each note in `notes`."""
    matrix = perspective_matrix(corners, warp_w, warp_h)
    out: List[str] = []
    for n in notes:
        fidx = int(round(n.onset * native_fps))
        frame = skeleton.get(str(fidx)) or {}
        key_x = pitch_to_x(n.pitch, warp_w)
        best, best_d = "unknown", float("inf")
        for hand in ("Left", "Right"):
            kpts = frame.get(hand)
            if not kpts:
                continue
            xs = _hand_points_strip(kpts, matrix, native_w, native_h)[:, 0]
            d = float(np.min(np.abs(xs - key_x)))
            if d < best_d:
                best_d, best = d, hand
        out.append(best)
    return out
