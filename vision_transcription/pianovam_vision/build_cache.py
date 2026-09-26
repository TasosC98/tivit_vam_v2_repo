"""Build the warped-strip cache (see ``strip_cache.py``) for a config's recordings.

One pass per video: decode sequentially, warp with the recording's keyboard
corners, JPEG-encode each 1408x112 strip. Resumable: recordings already cached
with matching settings are skipped, so just re-run the same command after an
interruption. Run on the server with the venv active:

  python -m pianovam_vision.build_cache --config configs/vam_full.yaml --workers 6
  python -m pianovam_vision.build_cache --config configs/yt_full.yaml  --workers 6

The cache directory is ``data.strip_cache`` from the config. Budget roughly
15-40 KB per frame (30 fps) -- the first finished recordings print their size,
so check ``df -h`` early.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "2048")

import numpy as np

from .config import load_config
from .metadata import filter_by_split, recordings_from_cfg
from .strip_cache import (
    expected_meta, try_open_cached, video_path_for, write_record_cache,
)


def _strips(reader, chunk: int, status: Dict[str, Any]):
    """Yield warped strips in order; stop cleanly at the first undecodable chunk."""
    n = len(reader)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        try:
            batch = reader.read_warped(np.arange(s, e))
        except Exception as ex:                      # corrupt tail -> truncate here
            status["error"] = f"decode failed at frame {s}/{n}: {ex}"
            return
        for fr in batch:
            yield fr


def build_one(cfg: Dict[str, Any], rec, quality: int, chunk: int) -> Dict[str, Any]:
    """Cache one recording. Runs inside a worker process."""
    import cv2
    from .video import WarpedVideo

    cv2.setNumThreads(1)                              # one core per worker
    t0 = time.time()
    kb = cfg["keyboard"]
    vp = video_path_for(cfg, rec)
    # Always cache colour, full length; grayscale / max_frames apply on read.
    reader = WarpedVideo(vp, rec.corners, kb["warp_width"], kb["warp_height"], False,
                         cfg["labels"]["fps"], 0, kb.get("decode_height", 0),
                         kb.get("read_chunk", 8))
    meta = expected_meta(cfg, rec)
    meta.update(record_time=rec.record_time, video=str(vp),
                video_bytes=vp.stat().st_size, native_fps=reader.native_fps,
                native_len=reader.native_len, expected_frames=len(reader))
    status: Dict[str, Any] = {}
    meta = write_record_cache(_strips(reader, chunk, status), cfg["data"]["strip_cache"],
                              rec.record_time, meta, quality)
    return {"record": rec.record_time, "frames": meta["num_frames"],
            "expected": len(reader), "mb": meta["bytes"] / 2**20,
            "sec": time.time() - t0, "error": status.get("error")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="splits to cache (default: the config's train+valid+test splits)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel processes (one video each); 1 = run inline")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    ap.add_argument("--chunk", type=int, default=64, help="frames decoded per read")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--limit", type=int, default=0, help="only the first N (testing)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    if not d.get("strip_cache"):
        raise SystemExit("set data.strip_cache in the config (or as an override)")
    splits = args.splits or (list(d["train_splits"]) + list(d["valid_splits"])
                             + list(d["test_splits"]))
    excl = set(d.get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), splits)
            if r.record_time not in excl]
    if args.limit:
        recs = recs[:args.limit]

    todo = []
    for r in recs:
        if not video_path_for(cfg, r).exists():
            print(f"  [skip] {r.record_time}: video missing")
            continue
        if not args.force and try_open_cached(cfg, r)[0] is not None:
            continue
        todo.append(r)
    print(f"cache: {d['strip_cache']}\n{len(recs)} recordings in {splits}: "
          f"{len(recs) - len(todo)} already cached / skipped, {len(todo)} to build "
          f"with {args.workers} worker(s)")
    if not todo:
        return

    Path(d["strip_cache"]).mkdir(parents=True, exist_ok=True)
    done = failed = 0
    tot_mb = tot_frames = 0.0
    t0 = time.time()

    def report(res) -> None:
        nonlocal done, tot_mb, tot_frames
        done += 1
        tot_mb += res["mb"]
        tot_frames += res["frames"]
        warn = f"  WARNING {res['error']}" if res["error"] else ""
        print(f"[{done + failed}/{len(todo)}] {res['record']}: {res['frames']} frames, "
              f"{res['mb']:.0f} MB, {res['sec']:.0f}s "
              f"({res['frames'] / max(res['sec'], 1e-6):.0f} fps) | total "
              f"{tot_mb / 1024:.1f} GB, {tot_mb * 1024 / max(tot_frames, 1):.1f} KB/frame"
              f"{warn}", flush=True)

    if args.workers <= 1:
        for r in todo:
            try:
                report(build_one(cfg, r, args.quality, args.chunk))
            except Exception as e:
                failed += 1
                print(f"[{done + failed}/{len(todo)}] {r.record_time}: FAILED ({e})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_one, cfg, r, args.quality, args.chunk): r for r in todo}
            for fut in as_completed(futs):
                try:
                    report(fut.result())
                except Exception as e:
                    failed += 1
                    print(f"[{done + failed}/{len(todo)}] {futs[fut].record_time}: "
                          f"FAILED ({e})", flush=True)

    hrs = tot_frames / cfg["labels"]["fps"] / 3600
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {done} built ({hrs:.1f} h of "
          f"video, {tot_mb / 1024:.1f} GB), {failed} failed. Failed recordings fall "
          f"back to on-the-fly decoding (or are skipped if the video is unreadable).")


if __name__ == "__main__":
    main()
