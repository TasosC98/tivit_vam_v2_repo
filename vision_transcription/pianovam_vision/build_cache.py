"""Build the warped-strip cache (see ``strip_cache.py``) for a config's recordings.

One pass per video: decode sequentially, warp with the recording's keyboard
corners, JPEG-encode each 1408x112 strip. Resumable: recordings already cached
with matching settings are skipped, so just re-run the same command after an
interruption. Run on the server with the venv active:

  python -m pianovam_vision.build_cache --config configs/vam_full.yaml --workers 6
  python -m pianovam_vision.build_cache --config configs/yt_full.yaml  --workers 6

The cache directory is ``data.strip_cache`` from the config. Budget roughly
15-40 KB per frame (30 fps) -- the first finished recordings print their size,
so check ``df -h`` early. The build stops starting new videos when the disk
gets down to ``--min_free_gb`` (videos in progress still finish, so keep that
margin larger than ``--workers`` x the biggest video, ~2 GB each).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "2048")

import numpy as np

from .config import load_config
from .metadata import filter_by_split, recordings_from_cfg
from .strip_cache import (
    expected_meta, failed_marker, remove_partial_files, try_open_cached,
    video_path_for, write_record_cache,
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


def build_isolated(args, record_time: str) -> Dict[str, Any]:
    """Build one recording in its own Python process.

    A video that crashes the decoder (segfault) or gets the process killed (out
    of memory) then costs only that video -- in a shared process pool one such
    crash fails every video still queued.
    """
    cmd = [sys.executable, "-m", "pianovam_vision.build_cache", "--config", args.config,
           "--only", record_time, "--quality", str(args.quality), "--chunk", str(args.chunk),
           *args.overrides]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=args.timeout_min * 60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"no result after {args.timeout_min} min (decoder stuck?)")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
            if "failed" in res:
                raise RuntimeError(res["failed"])
            return res
    if p.returncode < 0:
        sig = -p.returncode
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = "?"
        hint = {9: " -- killed, probably out of memory",
                11: " -- crash inside the video decoder"}.get(sig, "")
        raise RuntimeError(f"process died with signal {sig} ({name}){hint}")
    last = (p.stderr or "").strip().splitlines()[-1:] or ["no output"]
    raise RuntimeError(f"process exited with code {p.returncode}: {last[0][:300]}")


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
    ap.add_argument("--min_free_gb", type=float, default=30.0,
                    help="stop starting new videos below this much free disk space")
    ap.add_argument("--check", action="store_true",
                    help="only report whether the cache is complete for training "
                         "(train+valid splits); exit 3 if not")
    ap.add_argument("--timeout_min", type=float, default=120,
                    help="give up on a video after this many minutes")
    ap.add_argument("--only", default=None, help=argparse.SUPPRESS)  # internal: one video
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    if not d.get("strip_cache"):
        if args.check:
            return                                   # no cache configured: nothing to wait for
        raise SystemExit("set data.strip_cache in the config (or as an override)")

    if args.only:                                    # child process of build_isolated
        rec = next((r for r in recordings_from_cfg(cfg) if r.record_time == args.only), None)
        try:
            if rec is None:
                raise KeyError(f"no recording {args.only!r} in the metadata")
            res = build_one(cfg, rec, args.quality, args.chunk)
        except Exception as e:
            print("RESULT " + json.dumps({"record": args.only,
                                          "failed": f"{type(e).__name__}: {e}"}), flush=True)
            raise SystemExit(1)
        print("RESULT " + json.dumps(res), flush=True)
        return
    default_splits = list(d["train_splits"]) + list(d["valid_splits"])
    if not args.check:
        default_splits += list(d["test_splits"])
    splits = args.splits or default_splits
    excl = set(d.get("exclude_records", []) or [])
    recs = [r for r in filter_by_split(recordings_from_cfg(cfg), splits)
            if r.record_time not in excl]
    if args.limit:
        recs = recs[:args.limit]

    if args.check:
        missing, failed, no_video, ok = [], 0, 0, 0
        for r in recs:
            if not video_path_for(cfg, r).exists():
                no_video += 1
            elif try_open_cached(cfg, r)[0] is not None:
                ok += 1
            elif failed_marker(d["strip_cache"], r.record_time).exists():
                failed += 1                          # undecodable: training skips it too
            else:
                missing.append(r.record_time)
        print(f"strip cache {d['strip_cache']} ({'+'.join(splits)}): {ok} cached, "
              f"{failed} undecodable, {no_video} without video, {len(missing)} NOT cached yet")
        if missing:
            raise SystemExit(3)
        return

    if Path(d["strip_cache"]).is_dir():
        gb = remove_partial_files(d["strip_cache"])
        if gb > 0:
            print(f"removed partial files of an interrupted build ({gb:.1f} GB)")
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

    def free_gb() -> float:
        return shutil.disk_usage(d["strip_cache"]).free / 2**30

    def low_disk() -> bool:
        return free_gb() < args.min_free_gb

    if low_disk():
        print(f"only {free_gb():.0f} GB free on the cache disk "
              f"(< --min_free_gb {args.min_free_gb:g}): not starting", flush=True)
        raise SystemExit(2)
    done = failed = 0
    stopped_early = False
    tot_mb = tot_frames = 0.0
    t0 = time.time()

    def mark_failed(rid: str, err: Exception) -> None:
        failed_marker(d["strip_cache"], rid).write_text(str(err), encoding="utf-8")

    def report(res) -> None:
        nonlocal done, tot_mb, tot_frames
        done += 1
        tot_mb += res["mb"]
        tot_frames += res["frames"]
        failed_marker(d["strip_cache"], res["record"]).unlink(missing_ok=True)
        warn = f"  WARNING {res['error']}" if res["error"] else ""
        print(f"[{done + failed}/{len(todo)}] {res['record']}: {res['frames']} frames, "
              f"{res['mb']:.0f} MB, {res['sec']:.0f}s "
              f"({res['frames'] / max(res['sec'], 1e-6):.0f} fps) | total "
              f"{tot_mb / 1024:.1f} GB, {tot_mb * 1024 / max(tot_frames, 1):.1f} KB/frame"
              f"{warn}", flush=True)

    def disk_stop_message(queued: int) -> str:
        return (f"STOPPING: only {free_gb():.0f} GB free on the cache disk "
                f"(< --min_free_gb {args.min_free_gb:g}); {queued} video(s) not started. "
                f"Free some space (or lower --quality) and run the same command again.")

    if args.workers <= 1:
        for i, r in enumerate(todo):
            if low_disk():
                stopped_early = True
                print(disk_stop_message(len(todo) - i), flush=True)
                break
            try:
                report(build_one(cfg, r, args.quality, args.chunk))
            except Exception as e:
                failed += 1
                mark_failed(r.record_time, e)
                print(f"[{done + failed}/{len(todo)}] {r.record_time}: FAILED ({e})", flush=True)
    else:
        # Threads only wait; each video is decoded in its own process.
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_isolated, args, r.record_time): r for r in todo}
            for fut in as_completed(futs):
                if fut.cancelled():
                    continue
                try:
                    report(fut.result())
                except Exception as e:
                    failed += 1
                    mark_failed(futs[fut].record_time, e)
                    print(f"[{done + failed}/{len(todo)}] {futs[fut].record_time}: "
                          f"FAILED ({e})", flush=True)
                if not stopped_early and low_disk():
                    stopped_early = True
                    queued = sum(f.cancel() for f in futs)
                    print(disk_stop_message(queued) + " Waiting for the videos in "
                          "progress to finish...", flush=True)

    hrs = tot_frames / cfg["labels"]["fps"] / 3600
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {done} built ({hrs:.1f} h of "
          f"video, {tot_mb / 1024:.1f} GB), {failed} failed. Failed videos are skipped "
          f"by training and evaluation (reason in <cache>/<id>.failed); running this "
          f"command again retries them.")
    if stopped_early:
        print(f"INCOMPLETE: stopped for disk space ({free_gb():.0f} GB free). "
              f"Re-run the same command after freeing space; built videos are kept.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
