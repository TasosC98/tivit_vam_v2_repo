#!/usr/bin/env python
"""MediaPipe Hands landmarks for any piano video, in PianoVAM's Handskeleton layout.

Runs in its OWN environment (MediaPipe pins numpy and protobuf versions that the
training environment must not get):

    python3 -m venv ~/venvs/mediapipe
    ~/venvs/mediapipe/bin/pip install mediapipe==0.10.14

Needs only mediapipe, opencv and numpy; it does not import the project.

Output, one JSON per video (the layout of PianoVAM's ``Handskeleton/<rec>.json``,
so ``hands.read_skeleton_json`` reads it unchanged)::

    {"<frame>": {"Left": 21 x [x, y, z] | null, "Right": ...}, ...}

``frame`` = native frame index from 0, every frame of the video; x, y normalised
to the frame; z = MediaPipe's relative depth. "Left"/"Right" are MediaPipe's own
handedness on the UNFLIPPED frame, as in PianoVAM (for a top view they come out
mirrored; ``hands.swap_tags: true`` undoes that on reading). When MediaPipe gives
both hands the same tag, the more confident hand keeps it and the other gets the
other tag (counted as ``tag_conflicts``). Next to it ``<name>.meta.json``: the
settings, fps, frame size and how often each hand was found.

Commands
--------
extract   videos -> JSON, in parallel over videos, restartable (finished videos
          are skipped; ``--overwrite`` redoes them)
compare   our JSON against a reference JSON (PianoVAM's own) -> agreement
validate  which MediaPipe settings reproduce PianoVAM's landmarks: runs several
          settings on the first minute of a few recordings and compares each

    PY=~/venvs/mediapipe/bin/python
    $PY scripts/extract_landmarks.py validate --root /raid_storage/data_achatzigiannis/PianoVAM_v1.0
    $PY scripts/extract_landmarks.py extract /path/videos/*.mp4 --out_dir /path/landmarks --workers 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")       # MediaPipe's C++ info lines
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TAGS = ("Left", "Right")
WRIST, TIPS = 0, (4, 8, 12, 16, 20)
KEY_POINTS = (WRIST,) + TIPS                 # what the hand model reads

# Candidate settings for `validate` (name -> extract options). MediaPipe's defaults
# are video mode, model_complexity 1, confidences 0.5. Checked 2026-11-16 (mediapipe
# 0.10.14, first 20 s of the three VALIDATE_RECORDS): the defaults reproduce
# PianoVAM's Handskeleton (median 0.1-0.2 px, same hands found in 99-100 % of the
# frames, no tag swaps, in sync); complexity 0 and static mode differ by 2-6 px.
CANDIDATES: Dict[str, Dict] = {
    "video_c1": dict(static=False, complexity=1, det_conf=0.5, track_conf=0.5),
    "video_c0": dict(static=False, complexity=0, det_conf=0.5, track_conf=0.5),
    "static_c1": dict(static=True, complexity=1, det_conf=0.5, track_conf=0.5),
}
VALIDATE_RECORDS = ("2024-02-17_21-44-37", "2024-03-11_22-23-29", "2024-09-02_14-10-41")


# ------------------------------------------------------------------ extract
def _assign_tags(found: List[Tuple[str, float, list]]) -> Tuple[Dict[str, Optional[list]], int]:
    """MediaPipe's (label, score, points) per hand -> {"Left": pts|None, "Right": ...}."""
    out: Dict[str, Optional[list]] = {"Left": None, "Right": None}
    conflicts = 0
    for label, score, pts in sorted(found, key=lambda h: -h[1]):
        if out.get(label) is None:
            out[label] = pts
        else:
            other = "Right" if label == "Left" else "Left"
            conflicts += 1
            if out[other] is None:
                out[other] = pts
    return out, conflicts


def extract_video(video: str, out_dir: str, static: bool = False, complexity: int = 1,
                  det_conf: float = 0.5, track_conf: float = 0.5, max_frames: int = 0,
                  overwrite: bool = False, decimals: int = 6, log_every: int = 3000) -> Dict:
    import warnings
    warnings.filterwarnings("ignore", message=".*GetPrototype.*")   # protobuf, harmless
    import cv2
    import mediapipe as mp

    name = Path(video).name
    stem = name[:-4] if name.lower().endswith(".mp4") else Path(video).stem
    out = Path(out_dir) / f"{stem}.json"
    meta_path = Path(out_dir) / f"{stem}.meta.json"
    if out.exists() and meta_path.exists() and not overwrite:
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["skipped"] = True
        return m
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"video": video, "error": "cannot open the video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    hands = mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
                                     model_complexity=complexity,
                                     min_detection_confidence=det_conf,
                                     min_tracking_confidence=track_conf)
    tmp = out.with_name(out.name + ".partial")
    n = conflicts = 0
    present = np.zeros(2, np.int64)
    both = 0
    t0 = time.time()
    r = lambda v: round(float(v), decimals)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("{")
        while True:
            if max_frames and n >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            found = []
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    c = hd.classification[0]
                    found.append((c.label, float(c.score),
                                  [[r(p.x), r(p.y), r(p.z)] for p in lm.landmark]))
            entry, k = _assign_tags(found)
            conflicts += k
            got = [entry[t] is not None for t in TAGS]
            present += got
            both += all(got)
            f.write(("" if n == 0 else ", ") + json.dumps(str(n)) + ": " + json.dumps(entry))
            n += 1
            if log_every and n % log_every == 0:
                el = time.time() - t0
                print(f"  {stem}: {n}{'/' + str(total) if total else ''} frames, {n / el:.1f} fps", flush=True)
        f.write("}")
    cap.release()
    hands.close()
    el = time.time() - t0
    meta = {
        "video": str(video), "frames": n, "native_fps": fps, "size": size,
        "settings": dict(static=static, complexity=complexity, det_conf=det_conf,
                         track_conf=track_conf, max_frames=max_frames),
        "mediapipe": getattr(mp, "__version__", "?"), "seconds": round(el, 1),
        "fps_processed": round(n / el, 1) if el > 0 else None,
        "present": {t: round(float(present[i]) / max(n, 1), 4) for i, t in enumerate(TAGS)},
        "both_present": round(both / max(n, 1), 4), "tag_conflicts": conflicts,
        "complete": not max_frames or n < max_frames,
    }
    os.replace(tmp, out)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def _videos(args_videos: Sequence[str], list_file: Optional[str]) -> List[str]:
    vids: List[str] = []
    for v in args_videos:
        hits = sorted(glob.glob(v))
        vids.extend(hits if hits else [v])
    if list_file:
        with open(list_file, "r", encoding="utf-8") as f:
            vids.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    seen, out = set(), []
    for v in vids:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def run_extract(videos: Sequence[str], out_dir: str, workers: int, **kw) -> List[Dict]:
    print(f"{len(videos)} video(s) -> {out_dir} ({workers} workers; settings {kw})", flush=True)
    results: List[Dict] = []
    if workers <= 1:
        for v in videos:
            results.append(_report(extract_video(v, out_dir, **kw)))
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(extract_video, v, out_dir, **kw): v for v in videos}
        for fu in as_completed(futs):
            try:
                results.append(_report(fu.result()))
            except Exception as e:                       # one bad video must not stop the rest
                results.append(_report({"video": futs[fu], "error": repr(e)}))
    return results


def _report(m: Dict) -> Dict:
    if "error" in m:
        print(f"FAILED {m['video']}: {m['error']}", flush=True)
    elif m.get("skipped"):
        print(f"done before: {m['video']}", flush=True)
    else:
        print(f"done {Path(m['video']).name}: {m['frames']} frames at {m['fps_processed']} fps, "
              f"left tag {100 * m['present']['Left']:.1f} % / right tag {100 * m['present']['Right']:.1f} % "
              f"of frames, tag conflicts {m['tag_conflicts']}", flush=True)
    return m


# ------------------------------------------------------------------ compare
def read_raw(path: str, max_frames: int = 0) -> np.ndarray:
    """(N, 2, 21, 3) with the tags AS STORED (0 = "Left", 1 = "Right"), NaN if absent."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    n = max(int(k) for k in raw) + 1 if raw else 0
    if max_frames:
        n = min(n, max_frames)
    out = np.full((n, 2, 21, 3), np.nan, np.float32)
    for k, fr in raw.items():
        i = int(k)
        if i >= n:
            continue
        for h, t in enumerate(TAGS):
            v = (fr or {}).get(t)
            if v:
                out[i, h] = np.asarray(v, np.float32)[:21, :3]
    return out


def compare(ours_path: str, ref_path: str, size: Sequence[int] = (1920, 1080),
            max_frames: int = 0, max_shift: int = 6) -> Dict:
    """Agreement of our landmarks with a reference, per frame and hand tag.

    dist = mean pixel distance over wrist + five fingertips (what the model reads);
    tag swaps = frames with both hands in both files where the swapped pairing is
    closer than the stored one; shift = frame offset that fits best (0 = in sync)."""
    a = read_raw(ours_path, max_frames)
    b = read_raw(ref_path, max_frames)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    px = np.asarray(size, np.float32)
    pa, pb = ~np.isnan(a[:, :, 0, 0]), ~np.isnan(b[:, :, 0, 0])

    def dist(x, y):                                   # (..., 21, 3) -> (...) px
        d = (x[..., KEY_POINTS, :2] - y[..., KEY_POINTS, :2]) * px
        return np.sqrt((d ** 2).sum(-1)).mean(-1)

    res: Dict = {"frames": int(n)}
    for h, t in enumerate(TAGS):
        both = pa[:, h] & pb[:, h]
        d = dist(a[both, h], b[both, h]) if both.any() else np.zeros(0)
        res[t] = {
            "ref_present": round(float(pb[:, h].mean()), 4), "ours_present": round(float(pa[:, h].mean()), 4),
            "agree_present": round(float((pa[:, h] == pb[:, h]).mean()), 4),
            "median_px": round(float(np.median(d)), 2) if len(d) else None,
            "mean_px": round(float(d.mean()), 2) if len(d) else None,
            "p90_px": round(float(np.percentile(d, 90)), 2) if len(d) else None,
            "within_5px": round(float((d < 5).mean()), 4) if len(d) else None,
        }
    two = pa.all(1) & pb.all(1)
    if two.any():
        same = dist(a[two, 0], b[two, 0]) + dist(a[two, 1], b[two, 1])
        swap = dist(a[two, 0], b[two, 1]) + dist(a[two, 1], b[two, 0])
        res["tag_swaps"] = round(float((swap < same).mean()), 4)
        res["frames_two_hands"] = int(two.sum())
    best = None
    for s in range(-max_shift, max_shift + 1):        # ours[t + s] vs ref[t]
        lo, hi = max(0, -s), min(n, n - s)
        if hi - lo < 10:
            continue
        aa, bb = a[lo + s:hi + s], b[lo:hi]
        m = ~np.isnan(aa[:, :, 0, 0]) & ~np.isnan(bb[:, :, 0, 0])
        if not m.any():
            continue
        med = float(np.median(dist(aa[m], bb[m])))
        if best is None or med < best[1]:
            best = (s, med)
    if best:
        res["best_shift"], res["best_shift_median_px"] = best[0], round(best[1], 2)
    return res


def print_compare(name: str, r: Dict) -> None:
    L, R = r["Left"], r["Right"]
    print(f"  {name:<28} frames {r['frames']:>6} | found (ours/ref): L {100 * L['ours_present']:5.1f}/{100 * L['ref_present']:5.1f}"
          f"  R {100 * R['ours_present']:5.1f}/{100 * R['ref_present']:5.1f} | same-found L {100 * L['agree_present']:5.1f} R {100 * R['agree_present']:5.1f}"
          f" | median px L {L['median_px']} R {R['median_px']} | <5 px L {_pct(L['within_5px'])} R {_pct(R['within_5px'])}"
          f" | tag swaps {_pct(r.get('tag_swaps'))} | best shift {r.get('best_shift')}", flush=True)


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def verdict(rows: Sequence[Dict]) -> str:
    """Same landmarks as the reference? (median < 3 px, same hands found in >= 97 %
    of frames, tags swapped in < 1 % of frames, in sync)."""
    ok = all(
        r["Left"]["median_px"] is not None and r["Right"]["median_px"] is not None
        and max(r["Left"]["median_px"], r["Right"]["median_px"]) < 3.0
        and min(r["Left"]["agree_present"], r["Right"]["agree_present"]) >= 0.97
        and (r.get("tag_swaps") or 0) < 0.01 and r.get("best_shift", 0) == 0
        for r in rows)
    return "REPRODUCES the reference" if ok else "differs from the reference"


# ------------------------------------------------------------------ validate
def run_validate(root: str, out_dir: str, records: Sequence[str], frames: int, workers: int,
                 candidates: Sequence[str]) -> None:
    root_p = Path(root)
    jobs = []
    for c in candidates:
        for rec in records:
            video = root_p / "Video" / f"{rec}.mp4"
            ref = root_p / "Handskeleton" / f"{rec}.json"
            if not video.exists() or not ref.exists():
                print(f"missing {video if not video.exists() else ref}; left out", flush=True)
                continue
            jobs.append((c, rec, str(video), str(ref)))
    print(f"validate: {len(jobs)} runs ({', '.join(candidates)} x {len(records)} recordings), "
          f"first {frames} frames each -> {out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(extract_video, v, str(Path(out_dir) / c), max_frames=frames, **CANDIDATES[c]): (c, rec)
                for c, rec, v, _ in jobs}
        for fu in as_completed(futs):
            _report(fu.result())
    print("\n=== agreement with PianoVAM's own landmarks (tags as stored; px in 1920 x 1080) ===", flush=True)
    for c in candidates:
        rows = []
        for cc, rec, v, ref in jobs:
            if cc != c:
                continue
            ours = Path(out_dir) / c / f"{rec}.json"
            if not ours.exists():
                continue
            r = compare(str(ours), ref, max_frames=frames)
            rows.append(r)
            print_compare(f"{c} {rec}", r)
        if rows:
            fps = []
            for cc, rec, v, ref in jobs:
                mp_ = Path(out_dir) / c / f"{rec}.meta.json"
                if cc == c and mp_.exists():
                    with open(mp_, "r", encoding="utf-8") as f:
                        fps.append(json.load(f).get("fps_processed") or 0)
            print(f"  -> {c}: {verdict(rows)}; speed {np.mean(fps):.1f} frames/s per worker\n", flush=True)


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="videos -> landmark JSON")
    e.add_argument("videos", nargs="*", help="video files or glob patterns")
    e.add_argument("--list", help="text file with one video path per line")
    e.add_argument("--out_dir", required=True)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--max_frames", type=int, default=0, help="0 = the whole video")
    e.add_argument("--static", action="store_true", help="detect in every frame (no tracking)")
    e.add_argument("--complexity", type=int, default=1, choices=(0, 1))
    e.add_argument("--det_conf", type=float, default=0.5)
    e.add_argument("--track_conf", type=float, default=0.5)
    e.add_argument("--overwrite", action="store_true")

    c = sub.add_parser("compare", help="our JSON vs a reference JSON")
    c.add_argument("--ours", required=True, nargs="+")
    c.add_argument("--ref", required=True, nargs="+", help="one reference per --ours file, same order")
    c.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    c.add_argument("--max_frames", type=int, default=0)

    v = sub.add_parser("validate", help="which settings reproduce PianoVAM's landmarks")
    v.add_argument("--root", required=True, help="PianoVAM root with Video/ and Handskeleton/")
    v.add_argument("--out_dir", default="/raid_storage/data_achatzigiannis/landmarks/validate")
    v.add_argument("--records", nargs="+", default=list(VALIDATE_RECORDS))
    v.add_argument("--frames", type=int, default=3600, help="first N frames of each (3600 = 1 min at 60 fps)")
    v.add_argument("--workers", type=int, default=9)
    v.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        vids = _videos(a.videos, a.list)
        if not vids:
            print("no videos given", file=sys.stderr)
            return 2
        res = run_extract(vids, a.out_dir, a.workers, static=a.static, complexity=a.complexity,
                          det_conf=a.det_conf, track_conf=a.track_conf, max_frames=a.max_frames,
                          overwrite=a.overwrite)
        bad = [r for r in res if "error" in r]
        print(f"\n{len(res) - len(bad)} of {len(res)} videos done" + (f", {len(bad)} FAILED" if bad else ""))
        return 1 if bad else 0
    if a.cmd == "compare":
        if len(a.ours) != len(a.ref):
            print("give one --ref per --ours file", file=sys.stderr)
            return 2
        rows = []
        for o, r in zip(a.ours, a.ref):
            row = compare(o, r, size=a.size, max_frames=a.max_frames)
            rows.append(row)
            print_compare(Path(o).name, row)
        print(f"  -> {verdict(rows)}")
        return 0
    run_validate(a.root, a.out_dir, a.records, a.frames, a.workers, a.candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
