"""Keyboard corners for R3, annotated by hand.

R3 gives one rotated rectangle per session. That is too coarse for a per-key
model, which reads key k from column k of the warped strip: R3E1 (official
boxes) predicted one key too low on 56 of 74 R3s test chunks and one key too
high on 16 of 107 R3x test chunks (``offset_scan``), while V2N (one global
model) was not affected. The fix gives R3 what PianoVAM has: four corners per
keyboard that follow its perspective, set by eye with the model's key grid on
the warped strip (``scripts/corner_tool.html``). No labels are used, and train,
valid and test sessions are annotated the same way.

1. server: one picture per session (median of frames from several chunks, so
   the hands disappear) + the official corners as a starting point:

     python -m pianovam_vision.corners export --config configs/r3.yaml --out ~/r3_corner_frames

2. PC: copy that folder, open scripts/corner_tool.html in a browser, choose the
   folder, set the corners, download r3_corners.json -> configs/r3_corners.json
3. ``data.r3_corners: configs/r3_corners.json`` makes the loader use them
   (``metadata.load_r3_corners``); every session must then be annotated.

Points are [A0 back, C8 back, C8 front, A0 front] in native frame pixels -- the
order of ``Recording.corners`` (strip: A0 left, key backs at the top).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import load_config
from .metadata import Recording, recordings_from_cfg
from .strip_cache import video_path_for


def session_background(videos: List[Path], per_chunk: int = 5) -> np.ndarray:
    """Per-pixel median of frames spread over the given videos (BGR uint8).

    The keyboard does not move, the hands do: the median shows the bare keys.
    OpenCV with seeking (HEVC-safe, a few hundred MB at most).
    """
    import cv2

    frames = []
    for v in videos:
        cap = cv2.VideoCapture(str(v), cv2.CAP_FFMPEG)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in np.linspace(0.1 * (n - 1), 0.9 * (n - 1), per_chunk).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
    if not frames:
        raise IOError(f"no frame could be read from {[str(v) for v in videos]}")
    return np.median(np.stack(frames), axis=0).round().astype(np.uint8)


def export(cfg: Dict, out: Path, chunks_per_session: int = 3, per_chunk: int = 5,
           quality: int = 92, only: List[str] | None = None) -> List[Dict]:
    """Write ``out/frames/<session>.jpg`` and ``out/sessions.json``."""
    import cv2

    cfg = copy.deepcopy(cfg)
    cfg["data"]["r3_corners"] = None                  # start from the OFFICIAL boxes
    by_session: Dict[str, List[Recording]] = defaultdict(list)
    for r in recordings_from_cfg(cfg):
        by_session[r.session].append(r)
    if only:
        by_session = {s: v for s, v in by_session.items() if s in set(only)}
    groups: Dict[tuple, int] = {}
    rows = []
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for i, session in enumerate(sorted(by_session)):
        recs = sorted(by_session[session], key=lambda r: int(r.record_time.rsplit("split", 1)[1]))
        pick = sorted({int(round(x)) for x in np.linspace(0, len(recs) - 1, chunks_per_session)})
        img = session_background([video_path_for(cfg, recs[k]) for k in pick], per_chunk)
        cv2.imwrite(str(out / "frames" / f"{session}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        official = np.asarray(recs[0].corners, dtype=np.float64).round(2)
        key = (recs[0].subset, tuple(official.ravel()))
        rows.append({"session": session, "subset": recs[0].subset,
                     "splits": sorted({r.split for r in recs}), "chunks": len(recs),
                     "group": groups.setdefault(key, len(groups)),
                     "official": official.tolist(), "image": f"frames/{session}.jpg",
                     "frame_size": [int(img.shape[1]), int(img.shape[0])]})
        print(f"[{i + 1}/{len(by_session)}] {session}: {len(recs)} chunks, median of "
              f"{len(pick)} x {per_chunk} frames, box group {rows[-1]['group']}", flush=True)
    rows.sort(key=lambda r: (r["group"], r["session"]))
    with open(out / "sessions.json", "w", encoding="utf-8") as f:
        json.dump({"format": "r3_sessions_v1",
                   "order": "A0 back, C8 back, C8 front, A0 front (native frame px)",
                   "sessions": rows}, f, indent=1)
    print(f"{len(rows)} sessions, {len(groups)} distinct official boxes -> {out}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="one keyboard picture per session + official corners")
    ex.add_argument("--config", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--chunks_per_session", type=int, default=3)
    ex.add_argument("--frames_per_chunk", type=int, default=5)
    ex.add_argument("--quality", type=int, default=92)
    ex.add_argument("--sessions", nargs="*", default=None, help="only these (testing)")
    ex.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    export(cfg, Path(args.out).expanduser(), args.chunks_per_session, args.frames_per_chunk,
           args.quality, args.sessions)


if __name__ == "__main__":
    main()
