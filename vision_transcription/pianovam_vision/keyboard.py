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

from . import N_KEYS, PITCH_MIN


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


def pitch_to_column(pitch: int, width: int) -> int:
    """Approximate horizontal centre (px) of a pitch in the rectified strip.

    Assumes the keyboard quad spans exactly the 88 keys edge-to-edge. Used for
    visualisation only; training does not depend on this being exact.
    """
    idx = int(np.clip(pitch - PITCH_MIN, 0, N_KEYS - 1))
    return int((idx + 0.5) / N_KEYS * width)
