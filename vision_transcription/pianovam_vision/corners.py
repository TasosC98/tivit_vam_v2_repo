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
   folder, set the corners, download r3_corners.json -> configs/r3_corners_hand.json
3. refine them (any machine with the exported folder; a few minutes):

     python -m pianovam_vision.corners refine --frames ~/r3_corner_frames \
         --corners configs/r3_corners_hand.json --out configs/r3_corners.json \
         --keep rach3_2021-08-19_a01

4. ``data.r3_corners: configs/r3_corners.json`` makes the loader use them
   (``metadata.load_r3_corners``); every session must then have corners.

Why step 3: hand-set quads had both ends right but the middle of the keyboard up to
~0.5 white keys off on 17 of 144 sessions (2026-11-20). A few px of error in a front
corner gives the homography a perspective ALONG the keyboard that the scene does not
have; the ends stay pinned and the middle moves. ``refine`` moves each session's
corners by a few px (median 1, at most ~8) so that the black keys and the white-key
gaps of its strip sit where they sit in the median session, over the whole keyboard.
Sessions whose picture is too covered (hands in all sampled frames) are kept as set
by hand (``--keep``).

Points are [A0 back, C8 back, C8 front, A0 front] in native frame pixels -- the
order of ``Recording.corners`` (strip: A0 left, key backs at the top).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

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


# ------------------------------------------------------------------------- refine
STRIP_W, STRIP_H = 1408, 112          # the R3 strip; the px constants below are tuned for it
_SHIFTS = np.arange(-12, 12.01, 0.5)  # px searched around each feature (~0.44 white keys)
_BLOCKS = ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 52))  # white keys


class KeyFeatures:
    """Where the model grid puts the black keys and the white-key gaps of a strip, and
    how far from there a warped strip shows them.

    Black keys: rows 0.40-0.58 of the strip (just before their front end, below most of
    what lies on the keys). White-key gaps: two heights, rows 0.66-0.76 and 0.86-0.96,
    so that the gaps must stay vertical (with one height a shear could stand in for a
    shift where hands hide keys).
    """

    def __init__(self, w: int = STRIP_W, h: int = STRIP_H):
        from .keyboard import key_geometry

        self.w, self.h, self.wk = w, h, w / 52.0
        black = np.array(sorted((x0 + x1) / 2 for kind, x0, _, x1, _ in key_geometry(w, h).values()
                                if kind == "black"))
        gaps = np.arange(1, 52) * self.wk
        self.nb, self.ng = len(black), len(gaps)
        self.x = np.r_[black, gaps, gaps]
        self.y = np.r_[np.full(self.nb, 0.49 * h), np.full(self.ng, 0.71 * h),
                       np.full(self.ng, 0.91 * h)]
        self._cols = np.arange(w, dtype=np.float64)

    def _dark(self, strip: np.ndarray, y0: float, y1: float, k: int) -> np.ndarray:
        from scipy.ndimage import median_filter

        g = strip[int(y0 * self.h):int(y1 * self.h)].astype(np.float64).mean(axis=(0, 2))
        return -(g - median_filter(g, k, mode="nearest"))

    def _shift(self, d: np.ndarray, centre: float, width: float) -> Tuple[float, float]:
        """Best shift of a dark box of ``width`` px (contrast with a 6 px ring around it)."""
        a = np.abs(self._cols[None, :] - (centre + _SHIFTS)[:, None])
        box = a <= width / 2
        ring = (a <= width / 2 + 6) & ~box
        q = (box * d).sum(1) / box.sum(1) - (ring * d).sum(1) / ring.sum(1)
        k = int(np.argmax(q))
        return float(_SHIFTS[k]), float(q[k])

    def measure(self, strip: np.ndarray) -> np.ndarray:
        """Shift (px, + = to the right) of every feature from its grid place; nan = not found."""
        out = np.full(len(self.x), np.nan)
        db = self._dark(strip, 0.40, 0.58, 61)
        for i in range(self.nb):
            s, q = self._shift(db, self.x[i], 0.58 * self.wk)
            if q > 5 and abs(s) < 11.5:
                out[i] = s
        for k, (y0, y1) in enumerate(((0.66, 0.76), (0.86, 0.96))):
            dg = self._dark(strip, y0, y1, 9)
            for i in range(self.ng):
                j = self.nb + k * self.ng + i
                s, q = self._shift(dg, self.x[j], 2)
                if q > 3 and abs(s) < 11.5:
                    out[j] = s
        return out

    def blocks(self, shifts: np.ndarray) -> np.ndarray:
        """Median shift per block of 8 white keys (about an octave), in white keys."""
        pos = self.x / self.wk
        res = []
        for a, b in _BLOCKS:
            v = shifts[(pos >= a) & (pos < b) & ~np.isnan(shifts)]
            res.append(np.median(v) / self.wk if len(v) >= 3 else np.nan)
        return np.array(res)


def homography64(corners: np.ndarray, w: int = STRIP_W, h: int = STRIP_H) -> np.ndarray:
    """Image -> strip homography of ``perspective_matrix``, in float64 (OpenCV's is
    float32, too coarse for the finite differences of the solver)."""
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    a, b = [], []
    for (x, y), (u, v) in zip(np.asarray(corners, dtype=np.float64), dst):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.r_[np.linalg.solve(np.array(a), np.array(b)), 1.0].reshape(3, 3)


def _warp(img: np.ndarray, corners: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    from .keyboard import perspective_matrix

    return cv2.warpPerspective(img, perspective_matrix(np.asarray(corners, np.float32), w, h),
                               (w, h), flags=cv2.INTER_LINEAR)


def refine_one(img: np.ndarray, corners: np.ndarray, target: np.ndarray, feats: KeyFeatures,
               rounds: int = 3, prior_px: float = 1.5, max_move: float = 10.0) -> np.ndarray:
    """Corners near ``corners`` whose strip shows the features at ``target`` (strip px).

    Each round: measure the features on the strip, map them back to the image, solve
    the corners by robust least squares (soft-L1, 2 px) with the given corners as prior
    (a move of ``prior_px`` costs as much as 1 px of feature error), moves bounded.
    """
    from scipy.optimize import least_squares

    q0 = np.asarray(corners, dtype=np.float64)
    q = q0.copy()
    for _ in range(rounds):
        sh = feats.measure(_warp(img, q, feats.w, feats.h))
        ok = ~np.isnan(sh) & ~np.isnan(target)
        if ok.sum() < 12:
            break
        p = np.c_[feats.x[ok] + sh[ok], feats.y[ok], np.ones(ok.sum())] @ np.linalg.inv(
            homography64(q, feats.w, feats.h)).T
        pts = np.c_[p[:, :2] / p[:, 2:], np.ones(ok.sum())]
        tgt = target[ok]

        def resid(v):
            s = pts @ homography64(v.reshape(4, 2), feats.w, feats.h).T
            return np.r_[s[:, 0] / s[:, 2] - tgt, (v - q0.ravel()) / prior_px]

        q = least_squares(resid, q.ravel(), loss="soft_l1", f_scale=2.0, diff_step=1e-5,
                          bounds=(q0.ravel() - max_move, q0.ravel() + max_move)).x.reshape(4, 2)
    return q


def refine_sessions(images: Mapping[str, np.ndarray], corners: Mapping[str, np.ndarray],
                    keep: Iterable[str] = (), w: int = STRIP_W, h: int = STRIP_H
                    ) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """Refine every session (except ``keep``) towards the median session's key pattern.

    Returns the new corners and one report row per session (largest corner move in px,
    per-block offsets from the median session before/after, in white keys).
    """
    feats = KeyFeatures(w, h)
    keep = set(keep)
    before = {s: feats.measure(_warp(images[s], corners[s], w, h)) for s in sorted(corners)}
    typical = np.nanmedian(np.array(list(before.values())), axis=0)
    target = feats.x + typical
    new, rows = {}, []
    for s in sorted(corners):
        q0 = np.asarray(corners[s], dtype=np.float64)
        new[s] = q0.copy() if s in keep else refine_one(images[s], q0, target, feats)
        after = feats.measure(_warp(images[s], new[s], w, h))
        rows.append({"session": s, "kept": s in keep, "move_px": float(np.abs(new[s] - q0).max()),
                     "before": feats.blocks(before[s] - typical).round(3).tolist(),
                     "after": feats.blocks(after - typical).round(3).tolist()})
    return new, rows


def save_corners(path: Path, corners: Mapping[str, np.ndarray], **meta) -> None:
    """Write an ``r3_corners_v1`` file, one session per line."""
    lines = ['{', ' "format": "r3_corners_v1",']
    lines += [f' {json.dumps(k)}: {json.dumps(v)},' for k, v in meta.items() if k != "format"]
    lines += [f' "n": {len(corners)},', ' "corners": {']
    names = sorted(corners)
    for i, s in enumerate(names):
        q = ", ".join(f"[{x:g}, {y:g}]" for x, y in np.round(np.asarray(corners[s], float), 2))
        lines.append(f'  {json.dumps(s)}: [{q}]' + ("," if i < len(names) - 1 else ""))
    Path(path).write_text("\n".join(lines + [' }', '}', '']), encoding="utf-8", newline="\n")


def refine(frames: Path, corners_in: Path, out: Path, keep: Iterable[str] = ()) -> List[Dict]:
    """``refine_sessions`` on the pictures of ``export`` (``frames/sessions.json``)."""
    import cv2

    src = json.loads(Path(corners_in).read_text(encoding="utf-8"))
    hand = {s: np.asarray(q, dtype=np.float64) for s, q in src["corners"].items()}
    sessions = json.loads((Path(frames) / "sessions.json").read_text(encoding="utf-8"))["sessions"]
    images = {r["session"]: cv2.imread(str(Path(frames) / r["image"])) for r in sessions}
    missing = sorted(s for s in hand if images.get(s) is None)
    if missing:
        raise FileNotFoundError(f"no picture in {frames} for {missing}")
    keep = sorted(set(keep))
    unknown = [s for s in keep if s not in hand]
    if unknown:
        raise KeyError(f"--keep: no such session {unknown}")
    new, rows = refine_sessions(images, hand, keep)
    meta = {k: v for k, v in src.items() if k not in ("format", "n", "corners")}
    meta["refined"] = {"from": Path(corners_in).name, "by": "pianovam_vision.corners refine",
                       "kept_as_set_by_hand": keep}
    save_corners(out, new, **meta)
    split = {r["session"]: "/".join(r["splits"]) for r in sessions}
    worst = lambda b: float(np.nanmax(np.abs(b))) if np.isfinite(b).any() else float("nan")
    print(f"{'session':24} {'split':6} move px  offset per octave block vs the median session "
          f"(white keys): before -> after")
    for r in rows:
        if worst(np.array(r["before"])) > 0.2 or worst(np.array(r["after"])) > 0.2 or r["kept"]:
            print(f"{r['session']:24} {split.get(r['session'], ''):6} {r['move_px']:6.1f}   "
                  + " ".join(f"{v:+.2f}" for v in r["before"]) + "  ->  "
                  + " ".join(f"{v:+.2f}" for v in r["after"]) + ("   (kept)" if r["kept"] else ""))
    wb = np.array([worst(np.array(r["before"])) for r in rows])
    wa = np.array([worst(np.array(r["after"])) for r in rows])
    mv = np.array([r["move_px"] for r in rows])
    print(f"worst block per session: before median {np.nanmedian(wb):.2f} (> 0.3: {(wb > 0.3).sum()}), "
          f"after median {np.nanmedian(wa):.2f} (> 0.3: {(wa > 0.3).sum()}) of {len(rows)}; "
          f"corner moves median {np.median(mv):.1f} px, max {mv.max():.1f} px -> {out}")
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
    rf = sub.add_parser("refine", help="line the key pattern up along the whole keyboard")
    rf.add_argument("--frames", required=True, help="the folder written by export")
    rf.add_argument("--corners", required=True, help="hand-set corners (r3_corners_v1)")
    rf.add_argument("--out", required=True)
    rf.add_argument("--keep", nargs="*", default=[], help="sessions to keep as set by hand")
    args = ap.parse_args()
    if args.cmd == "refine":
        refine(Path(args.frames).expanduser(), Path(args.corners), Path(args.out), args.keep)
        return
    cfg = load_config(args.config, args.overrides)
    export(cfg, Path(args.out).expanduser(), args.chunks_per_session, args.frames_per_chunk,
           args.quality, args.sessions)


if __name__ == "__main__":
    main()
