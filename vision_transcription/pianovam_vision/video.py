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
        decode_height: int = 0,
        read_chunk: int = 8,
    ):
        import decord  # imported lazily so the package imports without decord

        self.path = str(video_path)
        # Probe native resolution once.
        probe = decord.VideoReader(self.path, num_threads=1)
        self.native_fps = float(probe.get_avg_fps()) or 60.0
        self.native_len = len(probe)
        nat_h, nat_w = probe[0].shape[:2]
        del probe

        # Optionally decode at reduced resolution (huge memory + speed win); the
        # keyboard corners are scaled to match the decoded frame size.
        sx = sy = 1.0
        if decode_height and decode_height < nat_h:
            dh = int(decode_height)
            dw = int(round(nat_w * dh / nat_h))
            self._vr = decord.VideoReader(self.path, num_threads=1, width=dw, height=dh)
            sx, sy = dw / nat_w, dh / nat_h
        else:
            self._vr = decord.VideoReader(self.path, num_threads=1)

        scaled_corners = corners.copy().astype(np.float32)
        scaled_corners[:, 0] *= sx
        scaled_corners[:, 1] *= sy
        self.matrix = keyboard.perspective_matrix(scaled_corners, warp_width, warp_height)
        self.warp_width = warp_width
        self.warp_height = warp_height
        self.grayscale = grayscale
        self.target_fps = target_fps
        self.read_chunk = max(1, int(read_chunk))

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
        """Return (T, H, W, C) uint8 warped strips for given target frame ids.

        Decoding is done in small chunks so peak memory stays low even for long
        clips (decoding e.g. 64 full-HD frames at once is hundreds of MB).
        """
        target_indices = np.asarray(target_indices, dtype=np.int64)
        native = self.native_indices(target_indices)
        T = len(native)
        out = np.empty(
            (T, self.warp_height, self.warp_width, 1 if self.grayscale else 3),
            dtype=np.uint8,
        )
        for s in range(0, T, self.read_chunk):
            e = min(s + self.read_chunk, T)
            batch = self._vr.get_batch(list(native[s:e])).asnumpy()  # (c,H,W,3)
            for j in range(e - s):
                out[s + j] = keyboard.warp_frame(
                    batch[j], self.matrix, self.warp_width, self.warp_height,
                    self.grayscale,
                )
            del batch
        return out
