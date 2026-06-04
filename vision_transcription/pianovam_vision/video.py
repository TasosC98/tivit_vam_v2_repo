"""Video reading + on-the-fly keyboard warping.

Uses decord for fast random access. Frames are subsampled from the native
60 fps down to ``labels.fps`` (the label/inference rate) and warped to the
rectified keyboard strip.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from . import keyboard


class WarpedVideo:
    """Random-access reader yielding warped keyboard strips at target fps."""

    def __init__(
        self,
        video_path: str | Path,
        corners: np.ndarray,
        warp_width: int,
        warp_height: int,
        grayscale: bool,
        target_fps: float,
        max_frames: int = 0,
    ):
        import decord  # imported lazily so the package imports without decord

        self.path = str(video_path)
        self._vr = decord.VideoReader(self.path, num_threads=1)
        self.native_fps = float(self._vr.get_avg_fps()) or 60.0
        self.native_len = len(self._vr)

        self.matrix = keyboard.perspective_matrix(corners, warp_width, warp_height)
        self.warp_width = warp_width
        self.warp_height = warp_height
        self.grayscale = grayscale
        self.target_fps = target_fps

        # Native frame index for each target frame (uniform subsampling).
        self.stride = max(1, int(round(self.native_fps / target_fps)))
        n = self.native_len // self.stride
        if max_frames > 0:
            n = min(n, max_frames)
        self.num_frames = int(n)
        self._native_index = (np.arange(self.num_frames) * self.stride).astype(np.int64)

    def __len__(self) -> int:
        return self.num_frames

    def native_indices(self, target_indices: np.ndarray) -> np.ndarray:
        idx = self._native_index[np.clip(target_indices, 0, self.num_frames - 1)]
        return np.clip(idx, 0, self.native_len - 1)

    def read_warped(self, target_indices: List[int] | np.ndarray) -> np.ndarray:
        """Return (T, H, W, C) uint8 warped strips for given target frame ids."""
        target_indices = np.asarray(target_indices, dtype=np.int64)
        native = self.native_indices(target_indices)
        batch = self._vr.get_batch(list(native)).asnumpy()  # (T, H, W, 3) RGB
        out = np.empty(
            (len(native), self.warp_height, self.warp_width,
             1 if self.grayscale else 3),
            dtype=np.uint8,
        )
        for i in range(len(native)):
            out[i] = keyboard.warp_frame(
                batch[i], self.matrix, self.warp_width, self.warp_height,
                self.grayscale,
            )
        return out
