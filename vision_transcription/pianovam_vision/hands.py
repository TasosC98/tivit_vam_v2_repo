"""Which hand, and which finger, pressed each note (PianoVAM only).

Data (PianoVAM v1.2, see HANDS.md)
----------------------------------
* ``Fingering/<rec>.tsv``     automatic labels for 106 recordings: the 5 TSV
  columns (row for row identical to ``TSV/``) + ``hand`` (L/R/Noinfo) +
  ``finger`` (1 = thumb .. 5 = little finger, or Noinfo). Made by the dataset
  authors from MediaPipe landmarks; 19.9 % of the notes are Noinfo.
* ``Fingering_GT/<rec>.tsv``  manual labels for the first 150 (one recording:
  300) notes of 11 recordings, 1,800 notes; the only real ground truth.
* ``Handskeleton/<rec>.json`` MediaPipe Hands, per NATIVE (60 fps) frame:
  ``{"<frame>": {"Left": 21 x [x, y, z] | null, "Right": ... }}`` with x, y
  normalised to the 1920 x 1080 frame. **The Left/Right tags are mirrored**
  (MediaPipe's selfie convention): the hand tagged "Right" is the pianist's
  left hand. Measured on the 1,800 manual labels: the nearest fingertip gives
  the correct hand for 97.7 % of the notes with the tags swapped, 2.3 % as
  stored. ``hands.swap_tags: true`` (the default) swaps them on reading.

Coordinates
-----------
Everything here works in the SAME rectified keyboard frame as the model's
input strip: the keyboard quadrilateral (metadata corners LT, RT, RB, LB) maps
to the unit square, x = 0 at the left end of A0 and 1 at the right end of C8,
y = 0 at the fallboard (where the black keys are) and 1 at the front edge of
the white keys (the pianist's side). A landmark is turned into key units with
the real key layout: 52 equal white keys, black keys 0.62 of a white key wide
and 0.62 of the key length long, centred on the boundary they straddle (as in
``keyboard.key_geometry``). This is what lets the per-key model (one ROI per
key) and the skeleton meet exactly at the key.

Classes
-------
A hand/finger label is one of 10 classes, ``L1..L5`` (0..4) and ``R1..R5``
(5..9); ``IGNORE`` (-1) marks an unknown label (Noinfo).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import N_KEYS, PITCH_MAX, PITCH_MIN
from .labels import Note

HANDS = ("L", "R")
HAND_NAMES = {"L": "left", "R": "right"}
FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}
N_CLASSES = 10
CLASS_NAMES = [f"{h}{f}" for h in HANDS for f in range(1, 6)]
IGNORE = -1

# MediaPipe Hands landmark ids: wrist and the five fingertips (thumb..little).
WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)
MODEL_LANDMARKS = (WRIST,) + FINGERTIPS        # what the model's skeleton input keeps

N_WHITE = 52
BLACK_WIDTH = 0.62        # of a white key (as keyboard.key_geometry)
BLACK_LENGTH = 0.62       # of the key length
# Length / width of a white key (~150 mm / 23.5 mm): converts a distance along
# the keys (strip y units) into white-key widths, so x and y distances add up.
KEY_ASPECT = 6.4

SKELETON_CACHE_VERSION = 1


# ----------------------------------------------------------------- classes
def class_id(hand: str, finger) -> int:
    """'L'/'R' + 1..5 -> 0..9; anything else (Noinfo, '') -> IGNORE."""
    if hand not in HANDS:
        return IGNORE
    try:
        f = int(finger)
    except (TypeError, ValueError):
        return IGNORE
    if not 1 <= f <= 5:
        return IGNORE
    return HANDS.index(hand) * 5 + f - 1


def class_hand(c: int) -> str:
    """'L', 'R', or '' for IGNORE."""
    return "" if c < 0 else HANDS[c // 5]


def class_finger(c: int) -> int:
    """1..5, or 0 for IGNORE."""
    return 0 if c < 0 else c % 5 + 1


# ----------------------------------------------------------------- labels
def fingering_path(cfg: Dict[str, Any], record_time: str, gt: bool = False) -> Path:
    d = cfg["data"]
    sub = d.get("fingering_gt_dir", "Fingering_GT") if gt else d.get("fingering_dir", "Fingering")
    return Path(d["root"]) / sub / f"{record_time}.tsv"


def read_fingering(path: str | Path, offset_field: str = "key_offset") -> List[Tuple[Note, int]]:
    """Read a Fingering(/_GT) TSV -> [(Note, class)] in FILE order.

    File order matters: Fingering_GT holds the first N rows of the recording's
    TSV, so its rows line up with the first N rows of ``Fingering/`` and ``TSV/``.
    """
    col = {"onset": 0, "key_offset": 1, "frame_offset": 2}
    off_idx = col[offset_field]
    out: List[Tuple[Note, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#") or line.startswith("onset"):
                continue
            p = line.split("\t")
            onset, offset = float(p[0]), float(p[off_idx])
            note = Note(onset, max(offset, onset), int(float(p[3])), int(float(p[4])))
            out.append((note, class_id(p[5].strip(), p[6].strip()) if len(p) >= 7 else IGNORE))
    return out


def build_hand_roll(labelled: Sequence[Tuple[Note, int]], num_frames: int, fps: float,
                    label_frames: int = 6, min_note_frames: int = 1) -> np.ndarray:
    """(num_frames, 88) int64 class targets, IGNORE where unknown.

    A labelled note marks its key from its onset frame for ``label_frames``
    frames (or until its key is released, if sooner): the finger is on the key
    then. Frame rounding is the same as ``labels.build_target_rolls``.
    """
    roll = np.full((num_frames, N_KEYS), IGNORE, dtype=np.int64)
    for n, c in sorted(labelled, key=lambda nc: (nc[0].onset, nc[0].pitch)):
        if c < 0 or not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        on_f = int(round(n.onset * fps))
        if on_f >= num_frames:
            continue
        off_f = max(int(round(n.offset * fps)), on_f + min_note_frames)
        on_f = max(0, on_f)
        end = min(off_f, on_f + max(1, int(label_frames)), num_frames)
        roll[on_f:max(end, on_f + 1), n.pitch - PITCH_MIN] = c
    return roll


# ----------------------------------------------------------------- geometry
def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 projective map taking the 4 ``src`` points onto the 4 ``dst`` points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.asarray(a), np.asarray(b))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (..., 2) points through a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64)
    q = pts @ m[:, :2].T + m[:, 2]
    return q[..., :2] / q[..., 2:3]


def strip_homography(corners: np.ndarray) -> np.ndarray:
    """Native pixels -> normalised strip coordinates (see module docstring)."""
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    return homography(np.asarray(corners, dtype=np.float64), unit)


def key_layout() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per key (A0..C8): centre x, half width (strip units) and length (y units)."""
    wk = 1.0 / N_WHITE
    cx = np.zeros(N_KEYS); hw = np.zeros(N_KEYS); ln = np.ones(N_KEYS)
    j = -1
    for k in range(N_KEYS):
        if (k + PITCH_MIN) % 12 in (1, 3, 6, 8, 10):          # black: on the boundary
            cx[k], hw[k], ln[k] = (j + 1) * wk, 0.5 * BLACK_WIDTH * wk, BLACK_LENGTH
        else:
            j += 1
            cx[k], hw[k] = (j + 0.5) * wk, 0.5 * wk
    return cx, hw, ln


def key_distance(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Distance (white-key widths) from points to key k's surface; 0 inside it."""
    cx, hw, ln = key_layout()
    dx = np.maximum(np.abs(x - cx[k]) - hw[k], 0.0) * N_WHITE
    dy = (np.maximum(y - ln[k], 0.0) + np.maximum(-y, 0.0)) * KEY_ASPECT
    return np.hypot(dx, dy)


# ----------------------------------------------------------------- skeleton
@dataclass
class SkeletonTrack:
    """Hand landmarks of one recording at the label frame rate, strip coordinates.

    pts     : (T, 2, 21, 3) float32 -- [x, y, z] per landmark; hand 0 = the
              pianist's LEFT hand, 1 = RIGHT (tags already swapped if needed)
    present : (T, 2) bool           -- MediaPipe found that hand in the frame
    """
    pts: np.ndarray
    present: np.ndarray

    def __len__(self) -> int:
        return len(self.present)

    def fit(self, num_frames: int) -> "SkeletonTrack":
        """Crop or pad (with absent hands) to exactly ``num_frames`` frames."""
        n = len(self)
        if n >= num_frames:
            return SkeletonTrack(self.pts[:num_frames], self.present[:num_frames])
        pad = num_frames - n
        return SkeletonTrack(
            np.concatenate([self.pts, np.zeros((pad,) + self.pts.shape[1:], np.float32)]),
            np.concatenate([self.present, np.zeros((pad, 2), bool)]))

    def shifted(self, frames: int) -> "SkeletonTrack":
        """Track whose frame k is this track's frame k + ``frames`` (absent outside)."""
        if not frames:
            return self
        T = len(self)
        src = np.arange(T) + int(frames)
        ok = (src >= 0) & (src < T)
        pts = np.zeros_like(self.pts)
        pres = np.zeros_like(self.present)
        pts[ok], pres[ok] = self.pts[src[ok]], self.present[src[ok]]
        return SkeletonTrack(pts, pres)

    def model_input(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), 2, 6, 4) float32 [x, y, z, present] for wrist + fingertips,
        zeros where a hand is absent (the model's ``skel`` input)."""
        idx = np.clip(np.asarray(idx, dtype=np.int64), 0, max(0, len(self) - 1))
        p = self.pts[idx][:, :, MODEL_LANDMARKS, :]                       # (T,2,6,3)
        pres = self.present[idx][:, :, None, None].astype(np.float32)
        out = np.concatenate([p * pres, np.broadcast_to(pres, p.shape[:3] + (1,))], axis=-1)
        return np.nan_to_num(out).astype(np.float32)


def skeleton_json_path(cfg: Dict[str, Any], record_time: str) -> Path:
    d = cfg["data"]
    return Path(d["root"]) / d.get("handskeleton_dir", "Handskeleton") / f"{record_time}.json"


def read_skeleton_json(path: str | Path, swap_tags: bool = True) -> np.ndarray:
    """(N, 2, 21, 3) float32 per native frame, NaN where a hand is missing;
    hand 0 = pianist's left. ``swap_tags`` undoes PianoVAM's mirrored tags."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frames = [int(k) for k in raw.keys()]
    n = max(frames) + 1 if frames else 0
    out = np.full((n, 2, 21, 3), np.nan, dtype=np.float32)
    tags = ("Right", "Left") if swap_tags else ("Left", "Right")
    for k, fr in raw.items():
        i = int(k)
        for h, tag in enumerate(tags):
            v = (fr or {}).get(tag)
            if v:
                out[i, h] = np.asarray(v, dtype=np.float32)[:21, :3]
    return out


def project_skeleton(raw: np.ndarray, corners: np.ndarray, native_size: Sequence[int],
                     native_fps: float, fps: float) -> SkeletonTrack:
    """Native-frame landmarks -> SkeletonTrack at ``fps`` in strip coordinates.

    Label frame k uses native frame round(k * native_fps / fps), the rule of
    ``video.plan_frame_sampling``, so landmarks and strips come from one frame.
    """
    ratio = float(native_fps) / float(fps)
    n = int(np.floor(len(raw) / ratio)) if len(raw) else 0
    src = np.clip(np.round(np.arange(n) * ratio).astype(np.int64), 0, max(0, len(raw) - 1))
    r = raw[src]                                                          # (T,2,21,3)
    present = ~np.isnan(r[:, :, 0, 0])
    px = r[..., :2] * np.asarray(native_size, dtype=np.float32)
    xy = apply_homography(strip_homography(corners), px).astype(np.float32)
    pts = np.concatenate([xy, r[..., 2:3]], axis=-1)
    pts[~present] = 0.0
    return SkeletonTrack(np.nan_to_num(pts).astype(np.float32), present)


def _skeleton_meta(cfg: Dict[str, Any], rec) -> Dict[str, Any]:
    h = cfg.get("hands") or {}
    return {
        "version": SKELETON_CACHE_VERSION,
        "fps": float(cfg["labels"]["fps"]),
        "native_fps": float(h.get("native_fps", 60.0)),
        "native_size": [int(v) for v in h.get("native_size", [1920, 1080])],
        "swap_tags": bool(h.get("swap_tags", True)),
        "corners": np.asarray(rec.corners, dtype=np.float64).round(3).tolist(),
    }


def skeleton_cache_path(cfg: Dict[str, Any], record_time: str) -> Optional[Path]:
    d = cfg["data"].get("skeleton_cache")
    return Path(d) / f"{record_time}.npz" if d else None


_SKEL_OFFSETS: Dict[str, Dict[str, int]] = {}


def skeleton_offset(cfg: Dict[str, Any], record_time: str) -> int:
    """Label frames the recording's landmarks are moved by (``data.skeleton_offsets``,
    a JSON ``{record_time: frames}`` written by ``check_hands --write_offsets``)."""
    path = cfg["data"].get("skeleton_offsets")
    if not path:
        return 0
    if path not in _SKEL_OFFSETS:
        with open(path, "r", encoding="utf-8") as f:
            _SKEL_OFFSETS[path] = {str(k): int(v) for k, v in json.load(f).items()}
    return _SKEL_OFFSETS[path].get(record_time, 0)


def load_skeleton(cfg: Dict[str, Any], rec, write_cache: bool = True,
                  apply_offset: bool = True) -> SkeletonTrack:
    """The recording's SkeletonTrack, aligned with its video (see ``skeleton_offset``).

    Read from ``data.skeleton_cache`` when it was built with the same settings,
    else parsed from the JSON (slow: 30 MB to 200 MB of text) and stored in the
    cache for next time. The cache holds the landmarks as published; the offset
    is applied on reading."""
    track = _load_skeleton_raw(cfg, rec, write_cache)
    return track.shifted(skeleton_offset(cfg, rec.record_time)) if apply_offset else track


def _load_skeleton_raw(cfg: Dict[str, Any], rec, write_cache: bool) -> SkeletonTrack:
    meta = _skeleton_meta(cfg, rec)
    cp = skeleton_cache_path(cfg, rec.record_time)
    if cp is not None and cp.exists():
        try:
            with np.load(cp) as z:
                if json.loads(str(z["meta"])) == meta:
                    return SkeletonTrack(z["pts"].astype(np.float32), z["present"].astype(bool))
        except Exception as e:                     # unreadable cache: rebuild it
            print(f"[hands] could not read {cp} ({e}); rebuilding it")
    raw = read_skeleton_json(skeleton_json_path(cfg, rec.record_time), meta["swap_tags"])
    track = project_skeleton(raw, rec.corners, meta["native_size"], meta["native_fps"], meta["fps"])
    if cp is not None and write_cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        # Unique per process: DataLoader workers may build the same file at once.
        tmp = cp.with_name(f"{cp.stem}.{os.getpid()}.partial.npz")
        np.savez_compressed(tmp, pts=track.pts.astype(np.float16), present=track.present,
                            meta=json.dumps(meta))
        tmp.replace(cp)
    return track


def alignment_curve(track: SkeletonTrack, notes: Sequence[Note], fps: float,
                    shifts: Sequence[int], max_notes: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """How well the landmarks fit the notes when landmark frame k + s is read for
    label frame k, for each shift s: (mean distance in white keys from the pressed
    key to the nearest fingertip one frame after the onset, clipped at 2; share of
    notes with a fingertip within 0.25 keys). Only notes seen at every shift
    count, so the shifts are compared on the same notes.

    Needs no hand labels: with landmarks in sync with the video the distance is
    smallest at the same small shift in every recording (a finger is on its key
    as it sounds); landmarks out of sync have their minimum elsewhere."""
    notes = [n for n in notes if PITCH_MIN <= n.pitch <= PITCH_MAX]
    if len(notes) > max_notes:
        notes = [notes[i] for i in np.linspace(0, len(notes) - 1, max_notes).astype(int)]
    nan = np.full(len(shifts), np.nan)
    if not notes or not len(track):
        return nan, nan
    cx, hw, ln = key_layout()
    k = np.array([n.pitch - PITCH_MIN for n in notes])
    f = np.array([int(round(n.onset * fps)) + 1 for n in notes])[:, None] + np.asarray(shifts)[None]
    ok = (f >= 0) & (f < len(track))
    fc = np.clip(f, 0, len(track) - 1)
    tips = track.pts[fc][:, :, :, FINGERTIPS, :2]                       # (N,S,2,5,2)
    pres = track.present[fc] & ok[..., None]                             # (N,S,2)
    kk = (slice(None), None, None, None)
    dx = np.maximum(np.abs(tips[..., 0] - cx[k][kk]) - hw[k][kk], 0) * N_WHITE
    dy = (np.maximum(tips[..., 1] - ln[k][kk], 0) + np.maximum(-tips[..., 1], 0)) * KEY_ASPECT
    d = np.where(pres[..., None], np.hypot(dx, dy), np.inf).min(axis=(2, 3))   # (N,S)
    both = pres.any(2).all(1)
    if not both.any():
        return nan, nan
    d = d[both]
    return np.minimum(d, 2.0).mean(0), (d < 0.25).mean(0)


ALIGN_SHIFTS = np.arange(-15, 16)


def choose_offset(dist: np.ndarray, ref_shift: int, min_frames: int = 2,
                  min_gain: float = 0.01, min_rel: float = 0.25) -> int:
    """Offset (frames) to apply, relative to ``ref_shift`` (where in-sync recordings
    have their minimum), from an ``alignment_curve`` distance over ALIGN_SHIFTS.
    0 unless the minimum is >= ``min_frames`` away AND clearly better: at least
    ``min_gain`` white keys and ``min_rel`` of the distance at ``ref_shift`` (a
    slow piece fits every shift almost equally; its minimum is noise)."""
    if np.isnan(dist).all():
        return 0
    b = int(np.nanargmin(dist))
    r = int(np.where(ALIGN_SHIFTS == ref_shift)[0][0])
    off = int(ALIGN_SHIFTS[b]) - int(ref_shift)
    gain = dist[r] - dist[b]
    if abs(off) < min_frames or gain < min_gain or gain < min_rel * dist[r]:
        return 0
    return off


# ----------------------------------------------------------------- deciding
def note_frames(n: Note, fps: float, label_frames: int, num_frames: int) -> np.ndarray:
    """Frames on which a note's hand/finger is read: its onset frame and up to
    ``label_frames - 1`` more while the key is held (as ``build_hand_roll``)."""
    on_f = int(round(n.onset * fps))
    off_f = max(int(round(n.offset * fps)), on_f + 1)
    end = min(off_f, on_f + max(1, int(label_frames)))
    f = np.arange(on_f, end)
    return f[(f >= 0) & (f < num_frames)]


def pitch_split(notes: Sequence[Note], split_pitch: int = 60) -> np.ndarray:
    """Baseline without vision: below ``split_pitch`` (C4 = 60) left, else right.
    Returns (N, 10) log-scores: certain about the hand, uniform over its fingers
    (so its finger decisions mean nothing; score it on the hand only)."""
    s = np.full((len(notes), N_CLASSES), -np.inf)
    for i, n in enumerate(notes):
        lo = 0 if n.pitch < split_pitch else 5
        s[i, lo:lo + 5] = np.log(0.2)
    return s


def fingertip_scores(notes: Sequence[Note], track: SkeletonTrack, fps: float,
                     label_frames: int = 6, roi: bool = True,
                     temperature: float = 0.1) -> np.ndarray:
    """Rule baseline from the skeleton: (N, 10) scores, higher = more likely.

    roi=True  ("roi" rule): each fingertip's distance to the pressed key's
              surface (0 inside it; real key shapes, white-key units), averaged
              over the note's frames (``note_frames``).
    roi=False ("tip" rule, the classic one): horizontal distance from each
              fingertip to the key centre at the onset frame only.
    Score = -distance / temperature (white keys): a small temperature makes the
    decision that of the nearest fingertip. -inf for a finger whose hand is not
    detected; a row of all -inf means no hand was seen (no decision possible).
    """
    cx, _, _ = key_layout()
    out = np.full((len(notes), N_CLASSES), -np.inf)
    T = len(track)
    for i, n in enumerate(notes):
        if not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        k = n.pitch - PITCH_MIN
        fr = note_frames(n, fps, label_frames if roi else 1, T)
        if len(fr) == 0:
            continue
        tips = track.pts[fr][:, :, FINGERTIPS, :]                       # (F,2,5,3)
        pres = track.present[fr]                                         # (F,2)
        if roi:
            d = key_distance(tips[..., 0], tips[..., 1], k)              # (F,2,5)
        else:
            d = np.abs(tips[..., 0] - cx[k]) * N_WHITE
        d = np.where(pres[:, :, None], d, np.nan)
        with np.errstate(invalid="ignore"):
            cnt = (~np.isnan(d)).sum(0)                                  # (2,5)
            mean = np.where(cnt > 0, np.nansum(d, 0) / np.maximum(cnt, 1), np.inf)
        out[i] = -mean.reshape(-1) / float(temperature)
    return out


def ayf_scores(notes: Sequence[Note], track: SkeletonTrack, fps: float,
               decay: float = 0.5, max_frames: int = 16) -> np.ndarray:
    """At Your Fingertips' rule (Moryossef et al., arXiv 2303.03745, Sec. III-E),
    re-implemented on our landmarks: (N, 10) log-probabilities.

    Per frame, every detected fingertip gets a Gaussian score of its horizontal
    position around the pressed key's centre with the key's width as standard
    deviation, normalised over all detected fingers; the per-frame distributions
    of the press frames are summed with exponentially declining weights (the
    first frame weighs most) and normalised. The paper gives neither the decay
    rate nor how a 2-D fingertip becomes a position on the key, so: horizontal
    position in the rectified strip, ``decay`` 0.5 per frame, at most
    ``max_frames`` frames while the key is held (later weights are < 1e-4).
    Decide with ``decide(..., joint=True)``: the paper takes the arg max over
    the 10 fingers."""
    cx, hw, _ = key_layout()
    out = np.full((len(notes), N_CLASSES), -np.inf)
    T = len(track)
    for i, n in enumerate(notes):
        if not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        k = n.pitch - PITCH_MIN
        fr = note_frames(n, fps, max_frames, T)
        if len(fr) == 0:
            continue
        x = track.pts[fr][:, :, FINGERTIPS, 0]                          # (F,2,5)
        pres = np.repeat(track.present[fr][:, :, None], 5, axis=2)       # (F,2,5)
        z = (x - cx[k]) / (2.0 * hw[k])
        logit = np.where(pres, -0.5 * z ** 2, -np.inf).reshape(len(fr), -1)   # (F,10)
        seen = np.isfinite(logit).any(1)
        if not seen.any():
            continue
        logit = logit[seen]
        p = np.exp(logit - logit.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w = float(decay) ** (fr[seen] - fr[0])
        q = (w[:, None] * p).sum(0) / w.sum()
        with np.errstate(divide="ignore"):
            out[i] = np.log(q)
    return out


def probs_scores(notes: Sequence[Note], probs: np.ndarray, fps: float,
                 label_frames: int = 6) -> np.ndarray:
    """(N, 10) log-probabilities of the model's hand/finger output (T, 88, 10),
    averaged over each note's frames; -inf rows where no frame is available."""
    out = np.full((len(notes), N_CLASSES), -np.inf)
    T = len(probs)
    for i, n in enumerate(notes):
        if not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        fr = note_frames(n, fps, label_frames, T)
        if len(fr):
            out[i] = np.log(np.clip(probs[fr, n.pitch - PITCH_MIN].mean(0), 1e-9, 1.0))
    return out


def chord_groups(notes: Sequence[Note], tol_s: float) -> List[List[int]]:
    """Indices of notes struck together: onsets chained within ``tol_s``."""
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset)
    groups: List[List[int]] = []
    for i in order:
        if groups and notes[i].onset - notes[groups[-1][-1]].onset <= tol_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """(N, 10) scores (log-probabilities up to a constant, -inf allowed) ->
    probabilities; rows without any finite score become all zeros."""
    s = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(s)
    ok = np.isfinite(s).any(1) if len(s) else np.zeros(0, bool)
    if ok.any():
        z = s[ok] - s[ok].max(1, keepdims=True)
        p = np.exp(np.where(np.isfinite(z), z, -np.inf))
        out[ok] = p / p.sum(1, keepdims=True)
    return out


def decide(notes: Sequence[Note], scores: np.ndarray, chord_matching: bool = False,
           chord_tol_s: float = 0.034, joint: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Turn (N, 10) scores into one class per note.

    The hand is decided first (larger total softmax mass over its 5 fingers),
    then the finger within that hand; ``joint`` takes the arg max over the 10
    classes instead (At Your Fingertips' rule). With ``chord_matching`` the notes of one
    chord (onsets within ``chord_tol_s``) get DIFFERENT fingers: the assignment
    with the best total score (Hungarian algorithm) -- one finger cannot press
    two keys at once, which independent per-key decisions cannot know.
    Returns (classes with IGNORE where no decision, confidence of the hand).
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    cls = np.full(n, IGNORE, dtype=np.int64)
    conf = np.zeros(n)
    ok = np.isfinite(s).any(1) if n else np.zeros(0, bool)
    if not ok.any():
        return cls, conf
    z = s[ok] - s[ok].max(1, keepdims=True)
    p = np.exp(np.where(np.isfinite(z), z, -np.inf))
    p /= p.sum(1, keepdims=True)
    pr = p[:, 5:].sum(1)
    if joint:
        best = p.argmax(1)
        hand, fing = best // 5, best % 5
    else:
        hand = (pr >= 0.5).astype(int)
        fing = np.where(hand == 1, p[:, 5:].argmax(1), p[:, :5].argmax(1))
    cls[ok] = hand * 5 + fing
    conf[ok] = np.where(hand == 1, pr, 1 - pr)
    if chord_matching:
        from scipy.optimize import linear_sum_assignment

        logp = np.full((n, N_CLASSES), -1e9)
        logp[ok] = np.log(np.clip(p, 1e-12, 1.0))
        for g in chord_groups(notes, chord_tol_s):
            g = [i for i in g if ok[i]]
            if len(g) < 2 or len(g) > N_CLASSES:
                continue
            rows, cols = linear_sum_assignment(-logp[g])
            for r, c in zip(rows, cols):
                cls[g[r]] = c
                ph = np.exp(logp[g[r], 5:]).sum()
                conf[g[r]] = ph if c >= 5 else 1 - ph
    return cls, conf


# ----------------------------------------------------------------- fingering prior
PRIOR_MAX_INTERVAL = 12          # semitones; larger jumps share the outermost bin


def hand_sequences(notes: Sequence[Note], hands: np.ndarray) -> List[np.ndarray]:
    """For the left (0) and right (1) hand: indices of its notes in playing
    order (onset, then pitch, so a chord is read upwards)."""
    order = sorted(range(len(notes)), key=lambda i: (notes[i].onset, notes[i].pitch))
    return [np.array([i for i in order if hands[i] == h], dtype=np.int64) for h in (0, 1)]


def _transition_bins(notes: Sequence[Note], idx: np.ndarray,
                     chord_tol_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """For each consecutive pair of ``idx``: struck together (1) or not, and the
    pitch-interval bin (0 .. 2 * PRIOR_MAX_INTERVAL)."""
    on = np.array([notes[i].onset for i in idx])
    pi = np.array([notes[i].pitch for i in idx])
    chord = (np.diff(on) <= chord_tol_s).astype(np.int64)
    d = np.clip(np.diff(pi), -PRIOR_MAX_INTERVAL, PRIOR_MAX_INTERVAL) + PRIOR_MAX_INTERVAL
    return chord, d.astype(np.int64)


@dataclass
class FingerPrior:
    """First-order fingering model, per hand (as in Nakamura et al.'s HMMs):
    P(finger of the hand's next note | its previous finger, the pitch interval,
    struck together or not). ``log_t``: (2 hands, 2, 2 * PRIOR_MAX_INTERVAL + 1,
    5 previous fingers, 5 next fingers), log-probabilities over the last axis.
    Learned from the TRAINING recordings' labels only (stored with the model)."""
    log_t: np.ndarray
    chord_tol_s: float = 0.034

    @classmethod
    def fit(cls, labelled: Sequence[Tuple[Sequence[Note], np.ndarray]],
            chord_tol_s: float = 0.034, alpha: float = 1.0) -> "FingerPrior":
        """``labelled``: (notes, classes) per recording; unlabelled notes (IGNORE)
        are skipped (the hand's sequence goes on with its next labelled note)."""
        c = np.full((2, 2, 2 * PRIOR_MAX_INTERVAL + 1, 5, 5), float(alpha))
        for notes, y in labelled:
            y = np.asarray(y)
            hands = np.where(y < 0, -1, y // 5)
            for h, idx in enumerate(hand_sequences(notes, hands)):
                if len(idx) < 2:
                    continue
                chord, d = _transition_bins(notes, idx, chord_tol_s)
                f = y[idx] % 5
                np.add.at(c, (h, chord, d, f[:-1], f[1:]), 1.0)
        return cls(np.log(c / c.sum(-1, keepdims=True)), float(chord_tol_s))

    def to_dict(self) -> Dict[str, Any]:
        return {"log_t": self.log_t.tolist(), "chord_tol_s": self.chord_tol_s}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FingerPrior":
        return cls(np.asarray(d["log_t"], dtype=np.float64), float(d["chord_tol_s"]))


def viterbi_fingers(notes: Sequence[Note], probs: np.ndarray, prior: FingerPrior,
                    weight: float = 1.0) -> np.ndarray:
    """Note-sequence decoding of the fingers: each note keeps the hand of its
    (N, 10) probabilities (larger mass); then, per hand, the finger sequence
    maximising  sum log p(finger | note) + weight * sum log prior(transition)
    (Viterbi). Rows of zeros (no evidence) stay IGNORE. Returns classes."""
    p = np.asarray(probs, dtype=np.float64)
    ok = p.sum(1) > 0
    hands = np.where(ok, (p[:, 5:].sum(1) >= 0.5).astype(np.int64), -1)
    cls = np.full(len(p), IGNORE, dtype=np.int64)
    for h, idx in enumerate(hand_sequences(notes, hands)):
        if len(idx) == 0:
            continue
        q = p[idx, 5 * h:5 * h + 5]
        em = np.log(np.clip(q / np.maximum(q.sum(1, keepdims=True), 1e-12), 1e-9, 1.0))
        chord, d = _transition_bins(notes, idx, prior.chord_tol_s)
        delta = em[0].copy()
        back = np.zeros((len(idx), 5), dtype=np.int64)
        for t in range(1, len(idx)):
            sc = delta[:, None] + weight * prior.log_t[h, chord[t - 1], d[t - 1]]
            back[t] = sc.argmax(0)
            delta = sc.max(0) + em[t]
        path = np.zeros(len(idx), dtype=np.int64)
        path[-1] = int(delta.argmax())
        for t in range(len(idx) - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        cls[idx] = 5 * h + path
    return cls


# ----------------------------------------------------------------- outputs
def write_hand_tsv(path: str | Path, notes: Sequence[Note], cls: np.ndarray,
                   conf: np.ndarray) -> None:
    """onset, offset, pitch, velocity, hand (L/R/?), finger (1-5/?), p_hand."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("onset\toffset\tnote\tvelocity\thand\tfinger\tp_hand\n")
        for n, c, p in zip(notes, cls, conf):
            f.write(f"{n.onset:.6f}\t{n.offset:.6f}\t{n.pitch}\t{n.velocity}\t"
                    f"{class_hand(int(c)) or '?'}\t{class_finger(int(c)) or '?'}\t{p:.3f}\n")


def write_hand_midi(path: str | Path, notes: Sequence[Note], cls: np.ndarray) -> None:
    """MIDI with one track per hand ("Left hand", "Right hand", plus "Undecided"
    when some notes have no decision). A score editor opens it as a two-staff
    piano part. MIDI has no field for fingers; they are in the TSV."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    tracks = {"L": pretty_midi.Instrument(0, name="Left hand"),
              "R": pretty_midi.Instrument(0, name="Right hand"),
              "": pretty_midi.Instrument(0, name="Undecided")}
    for n, c in zip(notes, cls):
        if n.offset <= n.onset:
            continue
        tracks[class_hand(int(c))].notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, n.velocity))), pitch=int(n.pitch),
            start=float(n.onset), end=float(n.offset)))
    for key in ("L", "R", ""):
        if tracks[key].notes or key:
            pm.instruments.append(tracks[key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
