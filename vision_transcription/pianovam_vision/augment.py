"""Training-time augmentation of warped keyboard clips (config block ``augment:``).

One random transform is drawn per CLIP and applied to all of its frames, so the
motion of a key press across frames is preserved. Two families:

* geometric -- small horizontal scale/shift, vertical shift and rotation of the
  strip. This imitates an imperfect keyboard crop (PianoYT's axis-aligned boxes,
  a different dataset's corner annotations), so the model cannot rely on a key
  sitting at an exact pixel column. Keep the shift below half a key column
  (``warp_width / 88`` px, i.e. 16 px at 1408) so labels stay correct.
* photometric -- brightness/contrast, occasional grayscale and pixel noise
  (lighting / camera / compression differences between videos and datasets).

Never flip horizontally: that would mirror the pitch axis.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def sample_affine(width: int, height: int, a: Dict[str, Any],
                  rng: np.random.Generator) -> np.ndarray:
    """2x3 matrix: rotate + horizontal scale about the centre, then shift."""
    s = 1.0 + rng.uniform(-1, 1) * float(a.get("hscale", 0.0))
    th = math.radians(rng.uniform(-1, 1) * float(a.get("rotate_deg", 0.0)))
    tx = rng.uniform(-1, 1) * float(a.get("hshift_px", 0.0))
    ty = rng.uniform(-1, 1) * float(a.get("vshift_px", 0.0))
    c, sn = math.cos(th), math.sin(th)
    cx, cy = width / 2.0, height / 2.0
    lin = np.array([[s * c, -sn], [s * sn, c]], dtype=np.float64)
    t = np.array([cx + tx, cy + ty]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float32)


def apply_geometric(frames: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Warp every (H, W, C) uint8 frame of a clip with the same 2x3 matrix."""
    import cv2

    T, H, W, C = frames.shape
    out = np.empty_like(frames)
    for i in range(T):
        w = cv2.warpAffine(frames[i], m, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        out[i] = w.reshape(H, W, C)
    return out


def apply_photometric(x, a: Dict[str, Any], rng: np.random.Generator):
    """In-place photometric jitter of a (T, C, H, W) float tensor in [0, 1]."""
    import torch

    b, c = float(a.get("brightness", 0.0)), float(a.get("contrast", 0.0))
    if b > 0 or c > 0:
        beta = 1.0 + rng.uniform(-1, 1) * b                  # brightness gain
        gamma = 1.0 + rng.uniform(-1, 1) * c                 # contrast gain
        mean = float(x.mean()) * beta
        x.mul_(beta * gamma).add_(mean * (1.0 - gamma))
    if x.shape[1] == 3 and rng.random() < float(a.get("gray_p", 0.0)):
        g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        x.copy_(g.unsqueeze(1).expand_as(x))
    std = float(a.get("noise_std", 0.0))
    if std > 0 and rng.random() < float(a.get("noise_p", 0.4)):
        gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        x.add_(torch.randn(x.shape, generator=gen) * std)
    return x.clamp_(0.0, 1.0)
