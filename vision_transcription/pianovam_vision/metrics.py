"""Evaluation metrics: frame-level (training monitor) and note-level (mir_eval)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .labels import Note


def frame_prf(pred: np.ndarray, target: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Binary precision/recall/F1 over a probability roll vs {0,1} target."""
    p = (pred >= thr).astype(np.float64)
    t = (target >= 0.5).astype(np.float64)
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def _to_intervals(notes: List[Note]):
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[n.onset, n.offset] for n in notes], dtype=np.float64)
    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    # mir_eval wants Hz.
    freqs = 440.0 * (2.0 ** ((pitches - 69) / 12.0))
    return intervals, freqs


def note_scores(
    ref: List[Note],
    est: List[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> Dict[str, float]:
    """Note-level F1 with mir_eval: onset-only and onset+offset."""
    import mir_eval

    ref_i, ref_f = _to_intervals(ref)
    est_i, est_f = _to_intervals(est)

    out: Dict[str, float] = {}
    if len(ref_i) == 0 or len(est_i) == 0:
        for k in ("onset_p", "onset_r", "onset_f1", "full_p", "full_r", "full_f1"):
            out[k] = 0.0
        return out

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_f, est_i, est_f,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )
    out.update(onset_p=p, onset_r=r, onset_f1=f)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_f, est_i, est_f,
        onset_tolerance=onset_tolerance, offset_ratio=offset_ratio,
    )
    out.update(full_p=p, full_r=r, full_f1=f)
    return out
