"""Perspective rectification of the keyboard region.

The 4 corner points in metadata bound the keyboard in the raw frame. We warp
that quadrilateral onto a fixed-size axis-aligned strip so that, across all
recordings, a given pitch lands in roughly the same horizontal position. The
model sees the whole strip and learns the pixel-region -> pitch mapping; the
helpers below are mainly for cropping/visualisation/debugging.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import N_KEYS, PITCH_MIN, PITCH_MAX


def perspective_matrix(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Matrix that maps the keyboard quad [LT, RT, RB, LB] -> [W x H] rect."""
    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst)


def warp_frame(
    frame: np.ndarray, matrix: np.ndarray, width: int, height: int, grayscale: bool
) -> np.ndarray:
    """Warp one HxWx3 (RGB) frame to the rectified strip.

    Returns uint8 array of shape (height, width, C) with C=1 if grayscale.
    """
    out = cv2.warpPerspective(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    if grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)[..., None]
    return out


# ----------------------------------------------------------------- key layout
# Real piano geometry: the 52 white keys tile the width uniformly; the 36 black
# keys sit ON the boundary between specific white keys, narrower and shorter.
# This is used for visualisation / sanity overlays only.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_WHITE_CLASSES = {0, 2, 4, 5, 7, 9, 11}   # C D E F G A B (semitone classes)


def is_white_key(pitch: int) -> bool:
    return (pitch % 12) in _WHITE_CLASSES


def note_name(pitch: int) -> str:
    """Scientific pitch name, e.g. 60 -> 'C4', 61 -> 'C#4', 21 -> 'A0'."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def white_key_pitches() -> list[int]:
    return [p for p in range(PITCH_MIN, PITCH_MAX + 1) if is_white_key(p)]


def pitch_to_x(pitch: int, width: int) -> int:
    """Accurate horizontal centre (px) of a key using the real white/black
    layout. White key -> centre of its slot; black key -> the boundary between
    the two white keys it straddles."""
    wp = white_key_pitches()
    wk = width / len(wp)
    wi = {p: i for i, p in enumerate(wp)}
    if is_white_key(pitch):
        return int(round((wi[pitch] + 0.5) * wk))
    j = wi[pitch - 1]                      # left white neighbour (always white)
    return int(round((j + 1) * wk))


def key_geometry(width: int, height: int) -> dict:
    """Return {pitch: (kind, x0, y0, x1, y1)} drawing boxes for all 88 keys.
    White keys are full-height slots; black keys are ~62% width/height, centred
    on the white boundary they straddle."""
    wp = white_key_pitches()
    wk = width / len(wp)
    wi = {p: i for i, p in enumerate(wp)}
    black_w = 0.62 * wk
    black_h = int(round(0.62 * height))
    boxes = {}
    for p in range(PITCH_MIN, PITCH_MAX + 1):
        if is_white_key(p):
            j = wi[p]
            boxes[p] = ("white", int(round(j * wk)), 0,
                        int(round((j + 1) * wk)), height)
        else:
            cx = (wi[p - 1] + 1) * wk
            boxes[p] = ("black", int(round(cx - black_w / 2)), 0,
                        int(round(cx + black_w / 2)), black_h)
    return boxes


def pitch_to_column(pitch: int, width: int) -> int:
    """Horizontal centre (px) of a pitch in the rectified strip (viz only).

    Now uses the real key layout (see ``pitch_to_x``); training does not depend
    on this. Kept for backward compatibility with existing callers.
    """
    return pitch_to_x(int(np.clip(pitch, PITCH_MIN, PITCH_MAX)), width)
