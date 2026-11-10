"""Convert predicted onset/frame probability rolls into note events.

Implements the standard Onsets-and-Frames decoding: a note begins on a rising
edge of the onset activation and is sustained while the frame activation stays
above threshold (or until the key is re-triggered).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import PITCH_MIN
from .labels import Note


def decode_notes(
    onset_probs: np.ndarray,        # (T, 88)
    frame_probs: np.ndarray,        # (T, 88)
    fps: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    min_duration_s: float = 0.03,
    velocity_probs: Optional[np.ndarray] = None,   # (T, 88) in [0,1] or None
    default_velocity: int = 80,
) -> List[Note]:
    """Decode notes from probability rolls.

    A note of key k starts at frame t when the onset probability crosses
    ``onset_threshold`` (o[t-1] < thr <= o[t]) and ends at the first later frame
    whose frame probability is below ``frame_threshold`` (onset frames count as
    active) or where a new onset of the same key starts. Its velocity is read at
    its onset frame (``default_velocity`` without a velocity head).
    """
    T, K = onset_probs.shape
    onset_bin = onset_probs >= onset_threshold
    frame_active = (frame_probs >= frame_threshold) | onset_bin

    notes: List[Note] = []
    for k in range(K):
        pitch = k + PITCH_MIN
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
    )
