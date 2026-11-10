"""Perspective rectification of the keyboard region.

The 4 corner points in metadata bound the keyboard in the raw frame. We warp
that quadrilateral onto a fixed-size axis-aligned strip so that, across all
recordings, a given pitch lands in roughly the same horizontal position: with a
1408-px strip, key k (counted from A0) occupies columns 16(k-1) .. 16k-1.
"""
from __future__ import annotations

import numpy as np


def perspective_matrix(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Matrix that maps the keyboard quad [LT, RT, RB, LB] -> [W x H] rect."""
    import cv2  # lazy: keeps the module importable (helpers, tests) without OpenCV

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
    import cv2  # lazy (see perspective_matrix)

    out = cv2.warpPerspective(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    if grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)[..., None]
    return out
