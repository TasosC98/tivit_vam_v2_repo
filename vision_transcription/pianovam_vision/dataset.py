"""Torch Dataset producing fixed-length clips of warped frames + target rolls.

Each item is a clip of ``clip_len`` frames from one recording:
    frames : (T, C, H, W) float32 in [0,1]
    frame  : (T, 88)      float32   key-held target
    onset  : (T, 88)      float32   onset target
    velocity:(T, 88)      float32   velocity/127 target (used only if enabled)

Frames come from the warped-strip cache when ``data.strip_cache`` is built
(fast), else from decoding the video on the fly. Readers are created lazily
inside each worker process so the Dataset stays picklable/fork-safe.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment as aug
from . import hands as hand_utils
from . import labels as label_utils
from .metadata import Recording
from .strip_cache import open_reader


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

        lab = cfg["labels"]
        self.fps = lab["fps"]
        self.onset_window = lab["onset_window_frames"]
        self.min_note_frames = lab["min_note_frames"]

        self.clip_len = cfg["train"]["clip_len"]
        self.clip_hop = cfg["train"]["clip_hop"] if train else self.clip_len
        self.max_frames = cfg["train"].get("max_frames_per_record", 0)

        # Clip augmentation (training only; see augment.py).
        a = cfg.get("augment") or {}
        self.aug = a if (train and a.get("enable", False)) else None

        # Per-worker lazy LRU caches. Capping open video readers is essential:
        # without it each persistent worker eventually opens every video at once
        # and the OS OOM-kills it.
        self.max_open_readers = cfg["train"].get("max_open_readers", 3)
        self.max_cached_targets = cfg["train"].get("max_cached_targets", 16)
        self._readers: "OrderedDict[str, Any]" = OrderedDict()
        self._targets: "OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        # Parsed label notes for every recording (small). Re-parsing a MIDI on
        # each target-roll cache miss is slow enough to bottleneck training.
        self._notes: Dict[str, List[label_utils.Note]] = {}

        # Hand/finger targets (+ skeleton input) for a model with a hand branch.
        # Off by default: then items, caches and random draws are exactly as before.
        mh = cfg["model"].get("hands") or {}
        hc = cfg.get("hands") or {}
        self.hands = bool(mh.get("enable", False))
        self.use_skel = self.hands and bool(mh.get("skeleton", True))
        self.hand_label_frames = int(hc.get("label_frames", 8))
        self.skel_drop_p = float(hc.get("skeleton_dropout", 0.1)) if train else 0.0
        self._hand_rolls: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._skeletons: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._no_fingering: set[str] = set()

        # Records whose video could not be decoded (logged once, then skipped).
        self._bad_records: set[str] = set()

        # Determine frame counts once (cheap header read) and build clip index.
        self._num_frames: Dict[str, int] = {}
        self.clips: List[Tuple[int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------ index
    def _build_index(self) -> None:
        sources: Counter = Counter()
        misses: Counter = Counter()
        for ri, rec in enumerate(self.recs):
            try:
                reader = self._open_reader(rec)
                n = len(reader)
            except Exception as e:  # corrupt/undecodable video -> exclude it
                self._bad_records.add(rec.record_time)
                print(f"[dataset] WARNING: excluding {rec.record_time} "
                      f"(cannot open video): {e}")
                continue
            sources[reader.source] += 1
            if getattr(reader, "cache_miss", None) and self.cfg["data"].get("strip_cache"):
                misses[reader.cache_miss.split(":")[0]] += 1
            # Skip recordings whose label file is absent (common for PianoYT,
            # where some YouTube videos/MIDIs are missing from the download).
            if not label_utils.reference_path(rec, self.cfg).exists():
                self._bad_records.add(rec.record_time)
                print(f"[dataset] WARNING: excluding {rec.record_time} "
                      f"(no label file)")
                continue
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
        kind = "train" if self.train else "eval"
        print(f"[dataset:{kind}] frames from strip cache: {sources['cache']} recording(s), "
              f"decoded from video: {sources['video']}"
              + (f" (cache misses: {dict(misses)} -> run build_cache)" if misses else ""))

    # ----------------------------------------------------------------- readers
    def _open_reader(self, rec: Recording):
        # Strip cache when built (fast), else on-the-fly decode + warp.
        return open_reader(self.cfg, rec, self.max_frames)

    def _get_reader(self, ri: int):
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
            notes = self._notes.get(rec.record_time)
            if notes is None:
                notes = label_utils.read_reference(rec, self.cfg)
                self._notes[rec.record_time] = notes
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

    def _lru(self, store: "OrderedDict", key: str, make):
        v = store.get(key)
        if v is None:
            v = make()
            store[key] = v
            while len(store) > self.max_cached_targets:
                store.popitem(last=False)
        else:
            store.move_to_end(key)
        return v

    def _get_hand_roll(self, ri: int) -> np.ndarray:
        """(n, 88) int8 hand/finger classes from Fingering/, IGNORE (-1) where unknown."""
        rec = self.recs[ri]
        n = self._num_frames[rec.record_time]

        def make():
            path = hand_utils.fingering_path(self.cfg, rec.record_time)
            if not path.exists():
                if rec.record_time not in self._no_fingering:
                    self._no_fingering.add(rec.record_time)
                    print(f"[dataset] WARNING: no fingering labels for {rec.record_time} "
                          f"({path}); its hand targets are all 'unknown'")
                lab = []
            else:
                lab = hand_utils.read_fingering(path, self.cfg["labels"]["offset_field"])
            roll = hand_utils.build_hand_roll(lab, n, self.fps, self.hand_label_frames,
                                              self.min_note_frames)
            return roll.astype(np.int8)
        return self._lru(self._hand_rolls, rec.record_time, make)

    def _get_skeleton(self, ri: int) -> np.ndarray:
        """(n, 2, 6, 4) float32 model input (wrist + fingertips per hand)."""
        rec = self.recs[ri]
        n = self._num_frames[rec.record_time]

        def make():
            track = hand_utils.load_skeleton(self.cfg, rec).fit(n)
            return track.model_input(np.arange(n))
        return self._lru(self._skeletons, rec.record_time, make)

    def _hand_items(self, ri: int, idx: np.ndarray, affine, width: int,
                    height: int) -> Dict[str, torch.Tensor]:
        out = {"hand": torch.from_numpy(self._get_hand_roll(ri)[idx].astype(np.int64))}
        if self.use_skel:
            sk = self._get_skeleton(ri)[idx].copy()                  # (T,2,6,4)
            if affine is not None:
                # The landmarks move with the strip: the same affine map, in pixels.
                px = np.stack([sk[..., 0] * width, sk[..., 1] * height, np.ones_like(sk[..., 0])], -1)
                q = px @ affine.T.astype(np.float32)
                pres = sk[..., 3] > 0
                sk[..., 0] = np.where(pres, q[..., 0] / width, 0.0)
                sk[..., 1] = np.where(pres, q[..., 1] / height, 0.0)
            if self.skel_drop_p > 0:
                # A hand MediaPipe misses: teach the model to fall back on the video.
                for h in range(2):
                    if np.random.random() < self.skel_drop_p:
                        sk[:, h] = 0.0
            out["skel"] = torch.from_numpy(sk)
        return out

    # -------------------------------------------------------------------- core
    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # A few videos fail to decode intermittently (slow EOF seeks / corrupt
        # files). Rather than crash a multi-hour run, skip a failed clip and
        # substitute another random one.
        last_err = None
        for _ in range(8):
            try:
                return self._load_clip(i)
            except Exception as e:
                last_err = e
                ri = self.clips[i][0]
                rt = self.recs[ri].record_time
                if rt not in self._bad_records:
                    self._bad_records.add(rt)
                    print(f"[dataset] WARNING: skipping unreadable clip from "
                          f"{rt}: {e}")
                self._readers.pop(rt, None)  # drop possibly-broken reader
                i = int(np.random.randint(0, len(self.clips)))
        raise RuntimeError(f"too many unreadable clips; last error: {last_err}")

    def _load_clip(self, i: int) -> Dict[str, torch.Tensor]:
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
        rng = affine = None
        if self.aug is not None and np.random.random() < float(self.aug.get("p", 0.8)):
            # Seeded from numpy's global RNG, which set_seed / DataLoader workers seed.
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            affine = aug.sample_affine(frames.shape[2], frames.shape[1], self.aug, rng)
            frames = aug.apply_geometric(frames, affine)
        strip_h, strip_w = frames.shape[1], frames.shape[2]
        frames = torch.from_numpy(frames).float().div_(255.0)
        frames = frames.permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)
        if rng is not None:
            frames = aug.apply_photometric(frames, self.aug, rng)

        froll, oroll, vroll = self._get_targets(ri)
        sl = slice(start, start + T)
        frame_t = torch.from_numpy(froll[sl].copy())
        onset_t = torch.from_numpy(oroll[sl].copy())
        vel_t = torch.from_numpy(vroll[sl].copy())

        item = {
            "frames": frames,
            "frame": frame_t,
            "onset": onset_t,
            "velocity": vel_t,
            "record_time": rec.record_time,
            "start": start,
        }
        if self.hands:
            item.update(self._hand_items(ri, idx, affine, strip_w, strip_h))
        return item
