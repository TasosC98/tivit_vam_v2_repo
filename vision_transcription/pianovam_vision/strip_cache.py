"""On-disk cache of warped keyboard strips (the model's exact input frames).

Decoding + warping a full-HD video on the fly is what makes training slow: every
training clip re-opens and seeks a 1080p video just to produce 32 small
1408x112 strips. The cache does that work ONCE per recording, sequentially
(the cheap way to decode), through the very same ``WarpedVideo`` path, and
stores each strip as a JPEG:

    <data.strip_cache>/<record_time>.bin       concatenated JPEG bytes
    <data.strip_cache>/<record_time>.idx.npy   int64 byte offsets (n+1,)
    <data.strip_cache>/<record_time>.json      settings the strips were built with

The ``.json`` is written last, so a half-written record is never used. Build
with ``python -m pianovam_vision.build_cache``; every reader of video frames
goes through ``open_reader``, which serves a recording from the cache when its
build settings match the config and falls back to decoding the video otherwise.

Strips are always cached in colour; a ``keyboard.grayscale: true`` config
converts on read, so one cache serves both variants.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

CACHE_VERSION = 1


def cache_paths(cache_dir: str | Path, record_time: str) -> Tuple[Path, Path, Path]:
    d = Path(cache_dir)
    return (d / f"{record_time}.bin", d / f"{record_time}.idx.npy",
            d / f"{record_time}.json")


def video_path_for(cfg: Dict[str, Any], rec) -> Path:
    d = cfg["data"]
    return rec.video_path(Path(d["root"]), d["video_dir"], d["video_ext"])


def expected_meta(cfg: Dict[str, Any], rec) -> Dict[str, Any]:
    """The build settings a cached recording must have to be usable by ``cfg``."""
    kb = cfg["keyboard"]
    return {
        "version": CACHE_VERSION,
        "warp_width": int(kb["warp_width"]),
        "warp_height": int(kb["warp_height"]),
        "fps": float(cfg["labels"]["fps"]),
        "decode_height": int(kb.get("decode_height", 0) or 0),
        "corners": np.asarray(rec.corners, dtype=np.float64).round(3).tolist(),
    }


def meta_mismatch(meta: Dict[str, Any], want: Dict[str, Any]) -> Optional[str]:
    """Reason the cached settings differ from ``want`` (None when they match)."""
    for k in ("version", "warp_width", "warp_height", "fps", "decode_height"):
        if meta.get(k) != want[k]:
            return f"{k}={meta.get(k)} (config wants {want[k]})"
    if not np.allclose(np.asarray(meta.get("corners", [])), np.asarray(want["corners"]),
                       atol=1e-2):
        return "keyboard corners changed"
    return None


def _decode_jpeg(buf: bytes, grayscale: bool) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("corrupt JPEG in strip cache")
    img = img[:, :, ::-1]                                   # BGR -> RGB
    if grayscale:
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)[..., None]
    return img


class CachedStrips:
    """Reader over one cached recording; same interface as ``WarpedVideo``."""

    source = "cache"

    def __init__(self, cache_dir: str | Path, record_time: str, grayscale: bool,
                 max_frames: int = 0):
        self.bin_path, idx_path, meta_path = cache_paths(cache_dir, record_time)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self._off = np.load(idx_path)
        n = len(self._off) - 1
        if max_frames and max_frames > 0:
            n = min(n, int(max_frames))
        self.num_frames = int(n)
        self.grayscale = bool(grayscale)
        self.warp_width = int(self.meta["warp_width"])
        self.warp_height = int(self.meta["warp_height"])

    def __len__(self) -> int:
        return self.num_frames

    def read_warped(self, target_indices) -> np.ndarray:
        """Return (T, H, W, C) uint8 RGB strips (C=1 if grayscale)."""
        idx = np.clip(np.asarray(target_indices, dtype=np.int64), 0, self.num_frames - 1)
        T = len(idx)
        out = np.empty((T, self.warp_height, self.warp_width, 1 if self.grayscale else 3),
                       dtype=np.uint8)
        off = self._off
        with open(self.bin_path, "rb") as f:
            i = 0
            while i < T:
                # Read each run of consecutive frames with a single seek + read.
                j = i
                while j + 1 < T and idx[j + 1] == idx[j] + 1:
                    j += 1
                base = int(off[idx[i]])
                f.seek(base)
                buf = f.read(int(off[idx[j] + 1]) - base)
                for k in range(i, j + 1):
                    s, e = int(off[idx[k]]) - base, int(off[idx[k] + 1]) - base
                    out[k] = _decode_jpeg(buf[s:e], self.grayscale)
                i = j + 1
        return out


def try_open_cached(cfg: Dict[str, Any], rec, max_frames: int = 0):
    """(reader, None) if ``rec`` is usable from the cache, else (None, reason)."""
    cache_dir = cfg["data"].get("strip_cache")
    if not cache_dir:
        return None, "no strip_cache configured"
    bin_p, idx_p, meta_p = cache_paths(cache_dir, rec.record_time)
    if not (meta_p.exists() and bin_p.exists() and idx_p.exists()):
        return None, "not cached"
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    why = meta_mismatch(meta, expected_meta(cfg, rec))
    if why:
        return None, f"stale cache: {why}"
    vp = video_path_for(cfg, rec)
    if vp.exists() and meta.get("video_bytes") not in (None, vp.stat().st_size):
        return None, "stale cache: video file changed since the cache was built"
    reader = CachedStrips(cache_dir, rec.record_time, cfg["keyboard"]["grayscale"], max_frames)
    return reader, None


def open_reader(cfg: Dict[str, Any], rec, max_frames: Optional[int] = None):
    """Frame reader for ``rec``: the strip cache if valid, else the video itself.

    ``max_frames`` defaults to ``train.max_frames_per_record`` (0 = whole video).
    The returned reader has ``.source`` = "cache" or "video"; a video fallback
    also carries ``.cache_miss`` (why the cache was not used).
    """
    from .video import WarpedVideo

    if max_frames is None:
        max_frames = cfg["train"].get("max_frames_per_record", 0)
    reader, why = try_open_cached(cfg, rec, max_frames)
    if reader is not None:
        return reader
    kb = cfg["keyboard"]
    reader = WarpedVideo(
        video_path_for(cfg, rec), rec.corners, kb["warp_width"], kb["warp_height"],
        kb["grayscale"], cfg["labels"]["fps"], max_frames,
        kb.get("decode_height", 0), kb.get("read_chunk", 8),
    )
    reader.source = "video"
    reader.cache_miss = why
    return reader


def write_record_cache(
    strips: Iterable[np.ndarray],
    cache_dir: str | Path,
    record_time: str,
    meta: Dict[str, Any],
    quality: int = 90,
) -> Dict[str, Any]:
    """Encode an iterable of (H, W, 3) RGB uint8 strips into the cache.

    Files are written under temporary names and renamed at the end (``.json``
    last), so an interrupted build never leaves a record that looks complete.
    Returns the metadata written.
    """
    import cv2

    bin_p, idx_p, meta_p = cache_paths(cache_dir, record_time)
    bin_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = {p: p.with_name(p.name + ".tmp") for p in (bin_p, idx_p, meta_p)}
    offsets = [0]
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    with open(tmp[bin_p], "wb") as f:
        for s in strips:
            ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(s[:, :, ::-1]), params)
            if not ok:
                raise IOError(f"JPEG encode failed for {record_time}")
            f.write(enc.tobytes())
            offsets.append(offsets[-1] + len(enc))
    if len(offsets) == 1:
        tmp[bin_p].unlink()
        raise IOError(f"no frames could be decoded for {record_time}")
    with open(tmp[idx_p], "wb") as f:
        np.save(f, np.asarray(offsets, dtype=np.int64))
    meta = dict(meta, num_frames=len(offsets) - 1, bytes=int(offsets[-1]),
                jpeg_quality=int(quality))
    with open(tmp[meta_p], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    for p in (bin_p, idx_p, meta_p):             # .json last = commit marker
        os.replace(tmp[p], p)
    return meta
