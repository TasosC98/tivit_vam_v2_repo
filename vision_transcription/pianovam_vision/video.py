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


def plan_frame_sampling(
    native_len: int, native_fps: float, target_fps: float, max_frames: int = 0
) -> np.ndarray:
    """Native frame indices for a uniform ``target_fps`` timeline.

    Target frame ``k`` maps to the native frame nearest to real time
    ``k / target_fps``, so frame ``k`` always corresponds to ~``k/target_fps``
    seconds **regardless of the video's native fps**. This keeps frames aligned
    with the labels and the decoder, which both operate at ``target_fps`` — the
    old integer-stride sampling only stayed aligned when native fps was an exact
    multiple of ``target_fps`` (true for PianoVAM's 60→30, but not for PianoYT's
    24/25/29.97 fps YouTube clips, which drifted out of sync).
    """
    native_fps = float(native_fps) or float(target_fps)
    ratio = native_fps / float(target_fps) if target_fps > 0 else 1.0
    if ratio <= 0:
        ratio = 1.0
    n = int(np.floor(native_len / ratio))
    n = max(0, n)
    if max_frames and max_frames > 0:
        n = min(n, int(max_frames))
    idx = np.round(np.arange(n) * ratio).astype(np.int64)
    if native_len > 0:
        idx = np.clip(idx, 0, native_len - 1)
    return idx


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

        # Native frame index for each target frame, sampled by timestamp so the
        # effective rate equals target_fps for any native fps (see docstring).
        self._native_index = plan_frame_sampling(
            self.native_len, self.native_fps, target_fps, max_frames
        )
        self.num_frames = int(len(self._native_index))

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


class SequentialWarpedVideo:
    """Front-to-back reader (OpenCV/FFmpeg) used to BUILD the strip cache.

    Same frames as ``WarpedVideo`` -- the same frame plan (``plan_frame_sampling``),
    the same corners and warp, RGB strips -- but each frame is decoded once, in
    order, without seeking, so memory stays at a few hundred MB. Chosen with
    ``data.cache_reader: cv2`` (R3). Why: decord needs ~7 GB of RAM per R3 video
    (1080p HEVC; 7.1 GB for a 3-min chunk, 6.9 GB for a 7.5-min one), so 8
    parallel builds failed on 373 of 895 videos, and on 217 of 372 when retried
    (out of memory, "no frames", EOF errors).
    """

    def __init__(self, video_path: str | Path, corners: np.ndarray, warp_width: int,
                 warp_height: int, target_fps: float, decode_height: int = 0):
        import cv2

        self.path = str(video_path)
        cap = self._open()
        try:
            self.native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
            self.native_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            nat_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            nat_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if self.native_len <= 0 or nat_w <= 0 or nat_h <= 0:
            raise IOError(f"cannot read the frame count / size of {self.path}")
        self.size = None                              # decode size (w, h) if resized
        sx = sy = 1.0
        if decode_height and decode_height < nat_h:
            dh = int(decode_height)
            dw = int(round(nat_w * dh / nat_h))
            self.size, sx, sy = (dw, dh), dw / nat_w, dh / nat_h
        scaled = np.asarray(corners, dtype=np.float32).copy()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy
        self.matrix = keyboard.perspective_matrix(scaled, warp_width, warp_height)
        self.warp_width, self.warp_height = warp_width, warp_height
        self._native_index = plan_frame_sampling(self.native_len, self.native_fps, target_fps)
        self.num_frames = int(len(self._native_index))

    def _open(self):
        import cv2

        params = []
        if hasattr(cv2, "CAP_PROP_N_THREADS"):        # one decoder thread per worker
            params = [cv2.CAP_PROP_N_THREADS, 1]
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            raise IOError(f"cannot open {self.path}")
        return cap

    def __len__(self) -> int:
        return self.num_frames

    def strips(self, status: dict | None = None):
        """Yield (H, W, 3) uint8 RGB strips in order, one per planned frame.

        If decoding ends before the plan does, stop there and put the reason in
        ``status["error"]`` (the cache then holds the frames decoded so far).
        """
        import cv2

        cap = self._open()
        pos, frame = -1, None
        try:
            for k, want in enumerate(self._native_index):
                while pos < want:
                    ok, frame = cap.read()
                    if not ok:
                        if status is not None:
                            status["error"] = (f"decode failed at frame {k}/{self.num_frames} "
                                               f"(video frame {pos + 1}/{self.native_len})")
                        return
                    pos += 1
                img = frame if self.size is None else cv2.resize(
                    frame, self.size, interpolation=cv2.INTER_AREA)
                # The warp treats channels alike: warp the BGR frame, then flip to RGB.
                out = cv2.warpPerspective(img, self.matrix, (self.warp_width, self.warp_height),
                                          flags=cv2.INTER_LINEAR)
                yield np.ascontiguousarray(out[:, :, ::-1])
        finally:
            cap.release()
