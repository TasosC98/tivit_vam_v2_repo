"""Evaluation metrics: frame-level (training monitor) and note-level (mir_eval)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def note_prf(
    ref: List[Note], est: List[Note], onset_tolerance: float = 0.05,
    offset_ratio: Optional[float] = None, offset_min_tolerance: float = 0.05,
) -> Tuple[float, float, float]:
    """Precision, recall and F1 exactly as mir_eval's precision_recall_f1_overlap.

    mir_eval only matches notes whose pitches are within 50 cents, i.e. the same
    MIDI pitch, so its maximum matching is the sum of one maximum matching per
    pitch. Matching pitch by pitch gives the same counts (so the same scores)
    without mir_eval's n_ref x n_est distance matrices, which for a long
    recording take gigabytes and minutes per call.
    """
    import mir_eval

    if not ref or not est:
        return 0.0, 0.0, 0.0
    ref_i, ref_f = _to_intervals(ref)
    est_i, est_f = _to_intervals(est)
    mir_eval.transcription.validate(ref_i, ref_f, est_i, est_f)
    by_ref, by_est = defaultdict(list), defaultdict(list)
    for n in ref:
        by_ref[n.pitch].append(n)
    for n in est:
        by_est[n.pitch].append(n)
    matched = 0
    for pitch in by_ref.keys() & by_est.keys():
        ri, rf = _to_intervals(by_ref[pitch])
        ei, ef = _to_intervals(by_est[pitch])
        matched += len(mir_eval.transcription.match_notes(
            ri, rf, ei, ef, onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance))
    precision = float(matched) / len(est)
    recall = float(matched) / len(ref)
    return precision, recall, mir_eval.util.f_measure(precision, recall)


def note_scores(
    ref: List[Note],
    est: List[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> Dict[str, float]:
    """Note-level F1 (mir_eval rules): onset-only and onset+offset."""
    p, r, f = note_prf(ref, est, onset_tolerance, offset_ratio=None)
    out: Dict[str, float] = dict(onset_p=p, onset_r=r, onset_f1=f)
    p, r, f = note_prf(ref, est, onset_tolerance, offset_ratio=offset_ratio)
    out.update(full_p=p, full_r=r, full_f1=f)
    return out


# Onset tolerances reported in the visual-transcription literature: 50 ms is the
# mir_eval / Onsets-and-Frames standard (V2N reports both); 100 ms is what
# "Pay Attention to the Keys" (PPAN) uses for its PianoYT numbers.
PAPER_TOLERANCES = (0.05, 0.10)


def note_scores_protocols(
    ref: List[Note], est: List[Note], tolerances=PAPER_TOLERANCES,
) -> Dict[str, float]:
    """Note metrics under every protocol used in the literature, per onset tolerance.

    For each tolerance tau (keys suffixed ``@50`` / ``@100`` in ms):
      onset_*@tau   pitch + onset within tau                 (S2S, V2R, PPAN, V2N "Onset")
      full_*@tau    + offset within max(20% of dur, 50 ms)   (mir_eval default, O&F)
      offtol_*@tau  + offset within tau                      (V2N "++Off")
    where * is p / r / f1.
    """
    out: Dict[str, float] = {}
    for tol in tolerances:
        ms = int(round(tol * 1000))
        variants = {
            "onset": dict(offset_ratio=None),
            "full": dict(offset_ratio=0.2, offset_min_tolerance=0.05),
            "offtol": dict(offset_ratio=0.0, offset_min_tolerance=tol),
        }
        for name, kw in variants.items():
            p, r, f = note_prf(ref, est, onset_tolerance=tol, **kw)
            out[f"{name}_p@{ms}"] = p
            out[f"{name}_r@{ms}"] = r
            out[f"{name}_f1@{ms}"] = f
    return out
