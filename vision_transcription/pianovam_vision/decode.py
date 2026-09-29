"""Convert predicted onset/frame probability rolls into note events.

Implements the standard Onsets-and-Frames decoding: a note begins on a rising
edge of the onset activation and is sustained while the frame activation stays
above threshold (or until the key is re-triggered).

Onset timing (``decode.onset_timing``, ``decode.time_shift_s``,
``decode.black_shift_s``) only moves notes in time; which notes exist is decided
on whole frames exactly as before, so the timing options never change the note
count. The defaults reproduce the original rule.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import PITCH_MIN
from .labels import Note

# Pitch classes of the black keys: C#, D#, F#, G#, A#.
BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
ONSET_TIMINGS = ("first", "centroid")


def _onset_centroid(p: np.ndarray, t: int, on: np.ndarray) -> float:
    """Probability-weighted mean frame of the onset peak that rises at frame t.

    The peak is the highest frame of the run above the threshold; the window is
    that peak's own hill, at most 2 frames each side (it stops where the
    probability rises again, so a neighbouring note on the same key is not
    mixed in). The result does not depend on which frame first crossed the
    threshold.
    """
    T = len(p)
    r = t
    while r < T and on[r]:
        r += 1                                   # frames t .. r-1 are above the threshold
    peak = t + int(np.argmax(p[t:r]))
    lo = peak
    while lo > max(0, peak - 2) and p[lo - 1] <= p[lo]:
        lo -= 1
    hi = peak
    while hi < min(T - 1, peak + 2) and p[hi + 1] <= p[hi]:
        hi += 1
    w = p[lo:hi + 1]
    s = float(w.sum())
    return float(np.dot(np.arange(lo, hi + 1), w) / s) if s > 0 else float(peak)


def decode_notes(
    onset_probs: np.ndarray,        # (T, 88)
    frame_probs: np.ndarray,        # (T, 88)
    fps: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    min_duration_s: float = 0.03,
    velocity_probs: Optional[np.ndarray] = None,   # (T, 88) in [0,1] or None
    default_velocity: int = 80,
    onset_timing: str = "first",
    time_shift_s: float = 0.0,
    black_shift_s: float = 0.0,
) -> List[Note]:
    """Decode notes from probability rolls.

    onset_timing: "first" puts the onset on the first frame above the threshold
      (whole frames, the classic rule). "centroid" puts it at the centre of the
      onset peak (sub-frame, see ``_onset_centroid``). With our 2-frame onset
      targets the centre sits about half a frame after the true onset, so
      calibrate ``time_shift_s`` together with it.
    time_shift_s: seconds added to every note (onset and offset): a timing bias of
      the model, or a label-vs-video offset between datasets.
    black_shift_s: seconds added on top for black keys, whose presses the camera
      sees later than those of white keys.
    """
    if onset_timing not in ONSET_TIMINGS:
        raise ValueError(f"onset_timing must be one of {ONSET_TIMINGS}, got {onset_timing!r}")
    T, K = onset_probs.shape
    onset_bin = onset_probs >= onset_threshold
    frame_active = (frame_probs >= frame_threshold) | onset_bin

    notes: List[Note] = []
    for k in range(K):
        pitch = k + PITCH_MIN
        shift = float(time_shift_s)
        if pitch % 12 in BLACK_PITCH_CLASSES:
            shift += float(black_shift_s)
        t = 0
        while t < T:
            rising = onset_bin[t, k] and (t == 0 or not onset_bin[t - 1, k])
            if not rising:
                t += 1
                continue

            off = t + 1
            while off < T and frame_active[off, k]:
                # Re-trigger: a new onset rising edge ends the current note.
                if onset_bin[off, k] and not onset_bin[off - 1, k]:
                    break
                off += 1

            onset_s = t / fps
            offset_s = off / fps
            if offset_s - onset_s >= min_duration_s:
                if velocity_probs is not None:
                    vel = int(np.clip(round(velocity_probs[t, k] * 127), 1, 127))
                else:
                    vel = default_velocity
                if onset_timing == "centroid":
                    onset_s = _onset_centroid(onset_probs[:, k], t, onset_bin[:, k]) / fps
                if shift:
                    onset_s += shift
                    offset_s += shift
                onset_s = max(onset_s, 0.0)
                offset_s = max(offset_s, onset_s)
                notes.append(Note(onset_s, offset_s, pitch, vel))
            t = off
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def decode_with_cfg(onset_probs: np.ndarray, frame_probs: np.ndarray,
                    velocity_probs: Optional[np.ndarray], cfg: Dict[str, Any]) -> List[Note]:
    """``decode_notes`` with every setting taken from the ``decode:`` config block."""
    d = cfg["decode"]
    return decode_notes(
        onset_probs, frame_probs, fps=cfg["labels"]["fps"],
        onset_threshold=d["onset_threshold"], frame_threshold=d["frame_threshold"],
        min_duration_s=d["min_duration_s"], velocity_probs=velocity_probs,
        default_velocity=d["default_velocity"],
        onset_timing=d.get("onset_timing", "first"),
        time_shift_s=float(d.get("time_shift_s", 0.0)),
        black_shift_s=float(d.get("black_shift_s", 0.0)),
    )


def describe_timing(d: Dict[str, Any]) -> str:
    """One-line summary of non-default timing settings ('' if all default)."""
    parts = []
    if d.get("onset_timing", "first") != "first":
        parts.append(f"onset timing {d['onset_timing']}")
    if float(d.get("time_shift_s", 0.0)):
        parts.append(f"shift {float(d['time_shift_s']) * 1000:+.0f} ms")
    if float(d.get("black_shift_s", 0.0)):
        parts.append(f"black keys {float(d['black_shift_s']) * 1000:+.0f} ms more")
    return ", ".join(parts)


def parse_grid(spec: str) -> List[float]:
    """"a:b:step" (inclusive) or "x,y,z" -> sorted list of floats."""
    spec = spec.strip()
    if ":" in spec:
        a, b, step = (float(x) for x in spec.split(":"))
        if step <= 0 or b < a:
            raise ValueError(f"bad grid {spec!r}: need start <= stop and step > 0")
        n = int(round((b - a) / step))
        vals = [round(a + i * step, 6) for i in range(n + 1)]
    else:
        vals = [float(x) for x in spec.split(",") if x.strip()]
    return sorted(set(vals))
