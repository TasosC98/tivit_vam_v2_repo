"""Torch Dataset producing fixed-length clips of warped frames + target rolls.

Each item is a clip of ``clip_len`` frames from one recording:
    frames : (T, C, H, W) float32 in [0,1]
    frame  : (T, 88)      float32   key-held target
    onset  : (T, 88)      float32   onset target
    velocity:(T, 88)      float32   velocity/127 target (used only if enabled)

Video readers (decord) are created lazily inside each worker process so the
Dataset stays picklable/fork-safe.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import labels as label_utils
from .metadata import Recording
from .video import WarpedVideo


class ClipDataset(Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        recordings: Sequence[Recording],
        train: bool,
    ):
        self.cfg = cfg
        self.recs: List[Recording] = list(recordings)
        self.train = train

        self.root = Path(cfg["data"]["root"])
        self.video_dir = cfg["data"]["video_dir"]
        self.tsv_dir = cfg["data"]["tsv_dir"]
        self.video_ext = cfg["data"]["video_ext"]

        kb = cfg["keyboard"]
        self.warp_w = kb["warp_width"]
        self.warp_h = kb["warp_height"]
        self.grayscale = kb["grayscale"]

        lab = cfg["labels"]
        self.fps = lab["fps"]
        self.offset_field = lab["offset_field"]
        self.onset_window = lab["onset_window_frames"]
        self.min_note_frames = lab["min_note_frames"]

        self.clip_len = cfg["train"]["clip_len"]
        self.clip_hop = cfg["train"]["clip_hop"] if train else self.clip_len
        self.max_frames = cfg["train"].get("max_frames_per_record", 0)

        # Per-worker lazy LRU caches. Capping open video readers is essential:
        # without it each persistent worker eventually opens every video at once
        # and the OS OOM-kills it.
        self.max_open_readers = cfg["train"].get("max_open_readers", 3)
        self.max_cached_targets = cfg["train"].get("max_cached_targets", 16)
        self._readers: "OrderedDict[str, WarpedVideo]" = OrderedDict()
        self._targets: "OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

        # Determine frame counts once (cheap header read) and build clip index.
        self._num_frames: Dict[str, int] = {}
        self.clips: List[Tuple[int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------ index
    def _build_index(self) -> None:
        for ri, rec in enumerate(self.recs):
            reader = self._open_reader(rec)
            n = len(reader)
            self._num_frames[rec.record_time] = n
            # Drop the reader created in the main process; workers reopen.
            self._readers.pop(rec.record_time, None)
            if n < 1:
                continue
            last_start = max(0, n - self.clip_len)
            starts = list(range(0, last_start + 1, self.clip_hop))
            if not starts:
                starts = [0]
            if starts[-1] != last_start:
                starts.append(last_start)
            for s in starts:
                self.clips.append((ri, s))

    # ----------------------------------------------------------------- readers
    def _open_reader(self, rec: Recording) -> WarpedVideo:
        vp = rec.video_path(self.root, self.video_dir, self.video_ext)
        return WarpedVideo(
            vp, rec.corners, self.warp_w, self.warp_h, self.grayscale,
            self.fps, self.max_frames,
        )

    def _get_reader(self, ri: int) -> WarpedVideo:
        rec = self.recs[ri]
        r = self._readers.get(rec.record_time)
        if r is None:
            r = self._open_reader(rec)
            self._readers[rec.record_time] = r
            while len(self._readers) > self.max_open_readers:
                self._readers.popitem(last=False)  # evict least-recently-used
        else:
            self._readers.move_to_end(rec.record_time)
        return r

    def _get_targets(self, ri: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rec = self.recs[ri]
        t = self._targets.get(rec.record_time)
        if t is None:
            notes = label_utils.read_tsv(
                rec.tsv_path(self.root, self.tsv_dir), self.offset_field
            )
            t = label_utils.build_target_rolls(
                notes, self._num_frames[rec.record_time], self.fps,
                self.onset_window, self.min_note_frames,
            )
            self._targets[rec.record_time] = t
            while len(self._targets) > self.max_cached_targets:
                self._targets.popitem(last=False)
        else:
            self._targets.move_to_end(rec.record_time)
        return t

    # -------------------------------------------------------------------- core
    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ri, start = self.clips[i]
        rec = self.recs[ri]
        n = self._num_frames[rec.record_time]
        T = self.clip_len

        if self.train and n > T:
            # Random temporal jitter around the indexed start.
            start = int(np.random.randint(0, n - T + 1))
        start = max(0, min(start, max(0, n - T)))
        idx = np.arange(start, start + T)
        idx = np.clip(idx, 0, n - 1)

        reader = self._get_reader(ri)
        frames = reader.read_warped(idx)                 # (T,H,W,C) uint8
        frames = torch.from_numpy(frames).float().div_(255.0)
        frames = frames.permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)

        froll, oroll, vroll = self._get_targets(ri)
        sl = slice(start, start + T)
        frame_t = torch.from_numpy(froll[sl].copy())
        onset_t = torch.from_numpy(oroll[sl].copy())
        vel_t = torch.from_numpy(vroll[sl].copy())

        return {
            "frames": frames,
            "frame": frame_t,
            "onset": onset_t,
            "velocity": vel_t,
            "record_time": rec.record_time,
            "start": start,
        }
