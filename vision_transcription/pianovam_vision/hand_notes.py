"""Hand and finger of every NOTE from the landmarks alone: the fast path.

The video model (vam_hands) and its skeleton-only ablation scored the same
(99.6 / 91.8 % on the 450 unseen manual notes): the hand/finger decision comes
from the landmarks. This module trains exactly that branch WITHOUT decoding a
single video frame, so a training run takes minutes instead of 6 hours, which
is what makes cross-validation, seeds and ablations affordable (HANDS.md, part 5).

Model (``NoteHandModel``)
-------------------------
1. Frame part = ``model.HandBranch`` with the skeleton input only (the same
   module and hyper-parameters as vam_hands_skel): for every frame and key, the
   12 landmarks measured relative to THAT key (``SkeletonKeyFeatures``), context
   across +/-30 keys and +/-2 frames, 10 classes L1..L5 / R1..R5. It is run on a
   short clip around each note's onset (onset - 2 .. onset + label_frames + 2
   frames) and read on the note's key over its first ``label_frames`` frames:
   the "frame read-out", identical to ``hands.probs_scores``.
2. Note part (new): one vector per note = the frame part's hidden feature at
   the note's key (mean over its read-out frames) + the note's pitch and timing
   context (``note_context``), then a 2-layer BiGRU over the notes in playing
   order -> 10 classes. This is the note-sequence model: it sees the hand and
   finger evidence of the neighbouring notes (a hand plays a line, a finger
   cannot strike two keys at once), which the frame part cannot.
Both outputs are trained together on the dataset's automatic labels (Noinfo
ignored) and both are saved, so every run gives the frame read-out and the
sequence model on the same notes. A third decoder needs no training:
``hands.viterbi_fingers`` with a ``FingerPrior`` learned from the training
labels (Nakamura-style fingering HMM), applied by hand_report.py.

Switches (``hand_notes:`` block):
  features     keyrel (ours) | strip (landmark coordinates in the rectified
               keyboard, not relative to the key) | image (raw image
               coordinates) | none (no vision at all: pitch + timing only, a
               learned MIDI-only hand separation baseline)
  sequence     true (BiGRU over notes) | false (per-note MLP)
  supervision  finger (10 classes) | hand (only which hand is supervised: does
               learning the fingers help the hand decision?)

Commands
--------
  # one model (default fold "official": train on train + ext-train, predict valid + test)
  python -m pianovam_vision.hand_notes train --config configs/hand_notes.yaml --out runs/hn/x [--fold F] [--seed 1] [k=v ...]
  # cross-validation: every fold of a scheme (and seeds), resumable
  python -m pianovam_vision.hand_notes cv --config configs/hand_notes.yaml --scheme loro --out runs/hn/loro_keyrel --seeds 1,2,3 [k=v ...]
  # the folds of a scheme, with sizes (no training)
  python -m pianovam_vision.hand_notes folds --config configs/hand_notes.yaml --scheme lopo
  # robustness: re-predict the held-out recordings of trained folds with corrupted landmarks
  python -m pianovam_vision.hand_notes robust --config configs/hand_notes.yaml --run_dir runs/hn/loro_keyrel \\
      [--tests corner:0.01,swap:0.05,...]
  # end to end: hands for transcribed notes (KeyWise / V2N MIDI), per recording
  python -m pianovam_vision.hand_notes e2e --config configs/hand_notes.yaml --model runs/hn/x/model.pt \
      --pred KeyWise=out/E1s2_vam_full_test --pred V2N=/path/V2N/pianovam_test --split test --out out/hn_e2e

Schemes (``folds``): official | loro (leave one manually labelled recording
out: 11 folds) | lopo (leave one pianist out: every recording of that pianist
leaves training; 7 folds) | session (leave one recording session = camera
set-up out: 3 folds). In the CV schemes the model trains on every other
usable recording (all splits; automatic labels only) for a fixed number of
steps, without model selection, and predicts the held-out recordings; the
manual labels are never used for training or selection anywhere.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import N_KEYS, PITCH_MAX, PITCH_MIN
from . import hands as H
from .config import load_config
from .labels import Note
from .metadata import Recording, recordings_from_cfg

NO_GT = -2                    # note without a manual label
N_SYM = 9                     # note_context features
ABS_DIM = 2 * len(H.MODEL_LANDMARKS) * 4     # strip / image features: [x, y, z, present] x 12
FEATURES = ("keyrel", "strip", "image", "none")
SCHEMES = ("official", "loro", "lopo", "session")
DEFAULT_SESSIONS = {"2024-02": "feb", "2024-03": "mar-apr", "2024-04": "mar-apr",
                    "2024-09": "sep"}


def hn_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("hand_notes") or {}


# ===================================================================== notes
def note_context(notes: Sequence[Note], chord_tol_s: float = 0.034) -> np.ndarray:
    """(N, N_SYM) pitch / timing context of notes in playing order, known for
    transcribed notes too: time to the previous / next onset, duration, chord
    size and rank, pitch relative to the chord, pitch, interval from the
    previous / to the next note."""
    n = len(notes)
    out = np.zeros((n, N_SYM), dtype=np.float32)
    if n == 0:
        return out
    on = np.array([x.onset for x in notes])
    p = np.array([x.pitch for x in notes], dtype=np.float64)
    dur = np.array([max(0.0, x.offset - x.onset) for x in notes])
    big = 4.0

    def logt(d):
        return np.log1p(np.minimum(np.maximum(d, 0.0), big) * 10) / np.log1p(big * 10)

    dprev = np.r_[big, np.diff(on)]
    dnext = np.r_[np.diff(on), big]
    size = np.ones(n)
    rank = np.full(n, 0.5)
    rel = np.zeros(n)
    for g in H.chord_groups(notes, chord_tol_s):
        g = sorted(g, key=lambda i: p[i])
        size[g] = len(g)
        if len(g) > 1:
            rank[g] = np.arange(len(g)) / (len(g) - 1)
        rel[g] = (p[g] - p[g].mean()) / 12.0
    out[:, 0] = logt(dprev)
    out[:, 1] = logt(dnext)
    out[:, 2] = logt(dur)
    out[:, 3] = np.minimum(size - 1, 8) / 4.0
    out[:, 4] = rank
    out[:, 5] = np.clip(rel, -3, 3)
    out[:, 6] = (p - 64.5) / 24.0
    out[:, 7] = np.clip(np.r_[0.0, np.diff(p)] / 12.0, -3, 3)
    out[:, 8] = np.clip(np.r_[np.diff(p), 0.0] / 12.0, -3, 3)
    return out


@dataclass
class NoteSet:
    """The notes of one recording (reference or transcribed) in playing order,
    with everything the model needs, and bookkeeping for the evaluation."""
    notes: List[Note]
    keys: np.ndarray                  # (N,) 0..87
    on_f: np.ndarray                  # (N,) onset frame
    n_read: np.ndarray                # (N,) read-out frames, 1..label_frames
    sym: np.ndarray                   # (N, N_SYM)
    y: np.ndarray                     # (N,) training target (automatic label), IGNORE if unknown
    skel: Optional[np.ndarray]        # (T, 2, 6, 4) landmarks in strip coordinates (model input)
    corners: np.ndarray               # (4, 2) keyboard corners, native pixels
    native_size: Tuple[int, int]
    rows: np.ndarray                  # (N,) index of each note in the source list (file row)
    auto: np.ndarray                  # (N,) dataset label
    gt: np.ndarray                    # (N,) manual label, NO_GT if none
    meta: Dict[str, Any] = field(default_factory=dict)
    track: Optional[H.SkeletonTrack] = None     # full landmarks (for the rules), eval sets only
    _s2px: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.notes)

    @property
    def strip_to_px(self) -> np.ndarray:
        if self._s2px is None:
            self._s2px = np.linalg.inv(H.strip_homography(self.corners))
        return self._s2px


def make_noteset(notes: Sequence[Note], cfg: Dict[str, Any], skel: Optional[np.ndarray],
                 corners: np.ndarray, auto: Optional[np.ndarray] = None,
                 gt: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None,
                 track: Optional[H.SkeletonTrack] = None) -> NoteSet:
    """Notes (any order) -> NoteSet in playing order (onset, pitch); notes outside
    the 88 keys are left out (none in PianoVAM)."""
    fps = float(cfg["labels"]["fps"])
    lf = int((cfg.get("hands") or {}).get("label_frames", 8))
    n_all = len(notes)
    auto = np.full(n_all, H.IGNORE, np.int64) if auto is None else np.asarray(auto, np.int64)
    gt = np.full(n_all, NO_GT, np.int64) if gt is None else np.asarray(gt, np.int64)
    keep = [i for i in range(n_all) if PITCH_MIN <= notes[i].pitch <= PITCH_MAX]
    order = np.array(sorted(keep, key=lambda i: (notes[i].onset, notes[i].pitch, i)), dtype=np.int64)
    ns = [notes[i] for i in order]
    on_f = np.array([int(round(x.onset * fps)) for x in ns], dtype=np.int64)
    off_f = np.array([max(int(round(x.offset * fps)), int(round(x.onset * fps)) + 1) for x in ns],
                     dtype=np.int64)
    n_read = np.clip(np.minimum(off_f, on_f + lf) - on_f, 1, lf)
    hn = hn_cfg(cfg)
    hc = cfg.get("hands") or {}
    return NoteSet(
        notes=ns, keys=np.array([x.pitch - PITCH_MIN for x in ns], dtype=np.int64),
        on_f=on_f, n_read=n_read.astype(np.int64),
        sym=note_context(ns, float(hn.get("chord_tol_s", 0.034))),
        y=auto[order].copy() if len(order) else np.zeros(0, np.int64),
        skel=skel, corners=np.asarray(corners, dtype=np.float64),
        native_size=tuple(int(v) for v in hc.get("native_size", [1920, 1080])),
        rows=order, auto=auto[order] if len(order) else np.zeros(0, np.int64),
        gt=gt[order] if len(order) else np.zeros(0, np.int64), meta=dict(meta or {}), track=track)


def session_of(cfg: Dict[str, Any], record_time: str) -> str:
    """Recording session (= camera set-up: the keyboard sits elsewhere in the
    image in each, see HANDS.md) from the date, ``hand_notes.sessions``."""
    s = dict(DEFAULT_SESSIONS)
    s.update(hn_cfg(cfg).get("sessions") or {})
    return s.get(record_time[:7], record_time[:7])


def load_noteset(cfg: Dict[str, Any], rec: Recording, keep_track: bool = False) -> NoteSet:
    """A recording's reference notes with automatic (+ manual) labels and landmarks.
    Raises FileNotFoundError when its labels or landmarks are missing."""
    from .evaluate_hands import load_references

    notes, auto, gt = load_references(cfg, rec)
    full_gt = np.full(len(notes), NO_GT, np.int64)
    if gt is not None:
        full_gt[:len(gt)] = gt
    track = H.load_skeleton(cfg, rec)
    skel = track.model_input(np.arange(len(track)))
    meta = {"record_time": rec.record_time, "split": rec.split, "performer": rec.performer,
            "session": session_of(cfg, rec.record_time), "n_manual": int(0 if gt is None else len(gt))}
    return make_noteset(notes, cfg, skel, rec.corners, auto, full_gt, meta,
                        track if keep_track else None)


# ===================================================================== inputs
def clip_input(ns: NoteSet, frames: np.ndarray, features: str,
               aug: Optional[Dict[str, Any]] = None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """(U, C, 2, 6, 4) landmarks [x, y, z, present] of the given frames (U, C);
    frames outside the recording have no hands. Training augmentation (one draw
    per call = per window): the strip's affine jitter of the video pipeline
    (augment block), keyboard-corner jitter, and dropping a hand. ``features``
    'image' turns x, y into image coordinates (0..1) at the end."""
    T = 0 if ns.skel is None else len(ns.skel)
    out = np.zeros(frames.shape + (2, 6, 4), dtype=np.float32)
    if T == 0:
        return out
    ok = (frames >= 0) & (frames < T)
    out[ok] = ns.skel[frames[ok]]
    pres = out[..., 3] > 0
    if aug is not None and rng is not None:
        if rng.random() < float(aug.get("p", 0.8)):
            from .augment import sample_affine

            W, Hh = float(aug["strip_w"]), float(aug["strip_h"])
            m = sample_affine(int(W), int(Hh), aug, rng).astype(np.float64)
            px = out[..., 0] * W
            py = out[..., 1] * Hh
            qx = m[0, 0] * px + m[0, 1] * py + m[0, 2]
            qy = m[1, 0] * px + m[1, 1] * py + m[1, 2]
            out[..., 0] = np.where(pres, qx / W, 0.0)
            out[..., 1] = np.where(pres, qy / Hh, 0.0)
        cj = float(aug.get("corner_jitter", 0.0))
        if cj > 0:
            width = float(np.linalg.norm(ns.corners[1] - ns.corners[0]))
            moved = ns.corners + rng.normal(0.0, cj * width, ns.corners.shape)
            m2 = H.strip_homography(moved) @ ns.strip_to_px
            xy = H.apply_homography(m2, out[..., :2])
            out[..., 0] = np.where(pres, xy[..., 0], 0.0)
            out[..., 1] = np.where(pres, xy[..., 1], 0.0)
        drop = float(aug.get("hand_dropout", 0.0))
        for h in range(2):
            if drop > 0 and rng.random() < drop:
                out[:, :, h] = 0.0
        pres = out[..., 3] > 0
    if features == "image":
        xy = H.apply_homography(ns.strip_to_px, out[..., :2]) / np.asarray(ns.native_size, np.float64)
        out[..., 0] = np.where(pres, xy[..., 0], 0.0)
        out[..., 1] = np.where(pres, xy[..., 1], 0.0)
    return out


# ===================================================================== model
def _torch():
    import torch
    return torch


def build_note_model(cfg: Dict[str, Any]):
    torch = _torch()
    nn = torch.nn
    from .model import HandBranch

    hn = hn_cfg(cfg)
    mh = dict(cfg["model"].get("hands") or {})
    hc = cfg.get("hands") or {}

    class NoteHandModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = str(hn.get("features", "keyrel"))
            if self.features not in FEATURES:
                raise ValueError(f"hand_notes.features must be one of {FEATURES}")
            self.L = int(hc.get("label_frames", 8))
            self.margin = int(mh.get("time_kernel", 5)) // 2
            if self.features == "keyrel":
                self.branch = HandBranch(0, {**mh, "video": False, "skeleton": True})
            elif self.features in ("strip", "image"):
                self.branch = HandBranch(ABS_DIM, {**mh, "video": True, "skeleton": False})
            else:
                self.branch = None
            d = int(mh.get("dim", 96)) if self.branch is not None else 0
            kd = int(hn.get("key_embed_dim", 16))
            nd = int(hn.get("note_dim", 128))
            drop = float(hn.get("dropout", 0.1))
            self.key_emb = nn.Embedding(N_KEYS, kd)
            self.note_in = nn.Sequential(nn.Linear(d + N_SYM + kd, nd), nn.ReLU(), nn.Dropout(drop))
            self.sequence = bool(hn.get("sequence", True))
            if self.sequence:
                layers = int(hn.get("seq_layers", 2))
                self.seq = nn.GRU(nd, nd // 2, num_layers=layers, batch_first=True,
                                  bidirectional=True, dropout=drop if layers > 1 else 0.0)
            else:
                self.seq = nn.Sequential(nn.Linear(nd, nd), nn.ReLU(), nn.Dropout(drop))
            self.note_out = nn.Linear(nd, H.N_CLASSES)
            # The note output REFINES the frame read-out: its logits are added to the
            # read-out's log-probabilities and start at zero, so training begins from
            # the frame decision and the sequence model learns corrections.
            self.residual = self.branch is not None and bool(hn.get("residual", True))
            if self.residual:
                nn.init.zeros_(self.note_out.weight)
                nn.init.zeros_(self.note_out.bias)

        @property
        def clip_len(self) -> int:
            return self.L + 2 * self.margin

        def frame_part(self, clips):
            """clips (M, C, 2, 6, 4) -> hidden (M, C, K, D), logits (M, C, K, 10)."""
            M, C = clips.shape[:2]
            if self.features == "keyrel":
                y = self.branch.encode(clips.new_zeros(M, C, N_KEYS, 0), clips)
            else:
                x, yy, z, p = clips.unbind(-1)
                yy = (2 * yy - 1) if self.features == "image" else yy.clamp(-1, 4) / 2
                a = torch.stack([(2 * x - 1) * p, yy * p, (10 * z).clamp(-1, 1) * p, p], -1)
                a = a.reshape(M, C, 1, ABS_DIM).expand(M, C, N_KEYS, ABS_DIM)
                y = self.branch.encode(a, None)
            return y, self.branch.out(self.branch.drop(y))

        def gather(self, y, logits, clip_of, key, n_read):
            """Each note's read-out frames on its key: z (B, N, D) mean hidden,
            frame logits (B, N, L, 10), mask (B, N, L)."""
            t = torch.arange(self.L, device=y.device) + self.margin
            ci, kk = clip_of[..., None], key[..., None]
            yn = y[ci, t, kk]                                         # (B,N,L,D)
            fl = logits[ci, t, kk]                                    # (B,N,L,10)
            mask = (torch.arange(self.L, device=y.device) < n_read[..., None]).to(yn.dtype)
            z = (yn * mask[..., None]).sum(2) / mask.sum(2, keepdim=True).clamp(min=1)
            return z, fl, mask

        @staticmethod
        def readout(fl, mask):
            """log of the frame probabilities averaged over the read-out frames (B, N, 10)."""
            p = torch.softmax(fl.float(), -1) * mask[..., None]
            return torch.log(p.sum(2) / mask.sum(2, keepdim=True).clamp(min=1) + 1e-6)

        def note_head(self, z, base, sym, key):
            parts = ([z] if z is not None else []) + [sym, self.key_emb(key)]
            h = self.note_in(torch.cat(parts, -1))
            h = self.seq(h)[0] if self.sequence else self.seq(h)
            out = self.note_out(h)
            return out + base if (self.residual and base is not None) else out

        def forward(self, clips, clip_of, key, n_read, sym):
            z = fl = mask = base = None
            if self.branch is not None:
                y, logits = self.frame_part(clips)
                z, fl, mask = self.gather(y, logits, clip_of, key, n_read)
                base = self.readout(fl, mask)
            return {"note_logits": self.note_head(z, base, sym, key), "frame_logits": fl, "frame_mask": mask}

    return NoteHandModel()


def class_nll(logits, y, supervision: str = "finger"):
    """Mean NLL over notes with a label (y >= 0). ``supervision`` 'hand' scores
    only the hand: -log of the summed probability of its five fingers."""
    torch = _torch()
    import torch.nn.functional as F

    valid = y >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    lp = F.log_softmax(logits[valid].float(), -1)
    yv = y[valid]
    if supervision == "hand":
        lh = torch.stack([torch.logsumexp(lp[:, :5], -1), torch.logsumexp(lp[:, 5:], -1)], -1)
        return F.nll_loss(lh, yv // 5)
    return F.nll_loss(lp, yv)


# ===================================================================== batches
def with_manual_labels(sets: Sequence[NoteSet]) -> Tuple[List[NoteSet], int]:
    """Training copies whose target is the MANUAL label wherever one exists (also
    on notes the dataset left without a label); the shared pool is not touched.
    Only ever called on training recordings, so the evaluated one's labels stay unseen."""
    out, n = [], 0
    for s in sets:
        m = s.gt >= 0
        if m.any():
            s = replace(s, y=np.where(m, s.gt, s.y))
            n += int(m.sum())
        out.append(s)
    return out, n


def sample_batch(sets: Sequence[NoteSet], weights: np.ndarray, n_windows: int, n_notes: int,
                 margin: int, clip_len: int, features: str, aug: Optional[Dict[str, Any]],
                 rng: np.random.Generator, manual_frac: float = 0.0) -> Dict[str, np.ndarray]:
    """Windows of ``n_notes`` consecutive notes from random training recordings;
    the notes of a window that start on the same frame share one clip.
    ``manual_frac`` > 0: that share of the windows contains a randomly chosen
    manually labelled note (the few manual notes are seen often enough to matter)."""
    clips, clip_of, keys, nread, sym, y = [], [], [], [], [], []
    m0 = 0
    man = [i for i, s in enumerate(sets) if (s.gt >= 0).any()] if manual_frac > 0 else []
    man_w = np.array([float((sets[i].gt >= 0).sum()) for i in man])
    for _ in range(n_windows):
        if man and rng.random() < manual_frac:
            s = sets[man[int(rng.choice(len(man), p=man_w / man_w.sum()))]]
            g = int(rng.choice(np.flatnonzero(s.gt >= 0)))
            st = int(np.clip(g - int(rng.integers(0, n_notes)), 0, len(s) - n_notes))
        else:
            s = sets[int(rng.choice(len(sets), p=weights))]
            st = int(rng.integers(0, len(s) - n_notes + 1))
        sl = slice(st, st + n_notes)
        uniq, inv = np.unique(s.on_f[sl], return_inverse=True)
        if features != "none":
            frames = uniq[:, None] - margin + np.arange(clip_len)[None]
            clips.append(clip_input(s, frames, features, aug, rng))
        clip_of.append(inv.reshape(-1) + m0)
        m0 += len(uniq)
        keys.append(s.keys[sl]); nread.append(s.n_read[sl]); sym.append(s.sym[sl]); y.append(s.y[sl])
    return {"clips": np.concatenate(clips) if clips else np.zeros((0, clip_len, 2, 6, 4), np.float32),
            "clip_of": np.stack(clip_of), "key": np.stack(keys), "n_read": np.stack(nread),
            "sym": np.stack(sym), "y": np.stack(y)}


def to_device(batch: Dict[str, np.ndarray], device) -> Dict[str, Any]:
    torch = _torch()
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    return out


# ===================================================================== predict
def predict_noteset(model, ns: NoteSet, device, clip_chunk: int = 256) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """(frame read-out probabilities or None, note-sequence probabilities), (N, 10) each."""
    torch = _torch()
    n = len(ns)
    if n == 0:
        z0 = np.zeros((0, H.N_CLASSES), np.float32)
        return (z0 if model.branch is not None else None), z0
    model.eval()
    keys = torch.from_numpy(ns.keys).to(device)
    sym = torch.from_numpy(ns.sym).to(device)
    z_all = p_frame = None
    with torch.no_grad():
        if model.branch is not None:
            uniq, inv = np.unique(ns.on_f, return_inverse=True)
            inv = inv.reshape(-1)
            D = int(model.branch.out.in_features)
            z_all = torch.zeros(n, D, device=device)
            p_frame = np.zeros((n, H.N_CLASSES), np.float32)
            for c0 in range(0, len(uniq), clip_chunk):
                cu = uniq[c0:c0 + clip_chunk]
                frames = cu[:, None] - model.margin + np.arange(model.clip_len)[None]
                clips = torch.from_numpy(clip_input(ns, frames, model.features)).to(device)
                y, logits = model.frame_part(clips)
                sel = np.where((inv >= c0) & (inv < c0 + len(cu)))[0]
                if not len(sel):
                    continue
                ci = torch.from_numpy(inv[sel] - c0).to(device)[None]
                z, fl, mask = model.gather(y, logits, ci, keys[sel][None],
                                           torch.from_numpy(ns.n_read[sel]).to(device)[None])
                z_all[sel] = z[0].float()
                pr = torch.softmax(fl[0].float(), -1) * mask[0][..., None]
                p_frame[sel] = (pr.sum(1) / mask[0].sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
        # note head over overlapping windows; each note weighted by how central it is
        W = min(int(hn_cfg_model(model).get("window_notes", 64)), n)
        hop = max(1, W // 2)
        starts = list(range(0, n - W + 1, hop))
        if starts[-1] != n - W:
            starts.append(n - W)
        acc = np.zeros((n, H.N_CLASSES), np.float64)
        wsum = np.zeros(n, np.float64)
        w = 1.0 - np.abs(np.arange(W) - (W - 1) / 2) / (W / 2) + 1e-3
        for st in starts:
            sl = slice(st, st + W)
            zz = z_all[sl][None] if z_all is not None else None
            base = (torch.log(torch.from_numpy(p_frame[sl]).to(device) + 1e-6)[None]
                    if p_frame is not None else None)
            lg = model.note_head(zz, base, sym[sl][None], keys[sl][None])[0]
            acc[sl] += w[:, None] * torch.softmax(lg.float(), -1).cpu().numpy()
            wsum[sl] += w
        p_seq = (acc / wsum[:, None]).astype(np.float32)
    return p_frame, p_seq


def hn_cfg_model(model) -> Dict[str, Any]:
    return getattr(model, "hn", {}) or {}


def load_note_model(path: str | Path, device="cpu"):
    torch = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_note_model(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.hn = hn_cfg(ck["cfg"])
    model.to(device).eval()
    prior = H.FingerPrior.from_dict(ck["prior"]) if ck.get("prior") else None
    return model, ck["cfg"], prior


# ===================================================================== baselines
def rule_scores(ns: NoteSet, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """(N, 10) scores of the rule baselines on the NoteSet's notes, + which hands
    MediaPipe saw at each onset frame."""
    hc = cfg.get("hands") or {}
    fps, lf = float(cfg["labels"]["fps"]), int(hc.get("label_frames", 8))
    out = {"s_pitch": H.pitch_split(ns.notes, int(hc.get("split_pitch", 60)))}
    tr = ns.track
    if tr is not None:
        out["s_tip"] = H.fingertip_scores(ns.notes, tr, fps, lf, roi=False)
        out["s_roi"] = H.fingertip_scores(ns.notes, tr, fps, lf, roi=True)
        out["s_ayf"] = H.ayf_scores(ns.notes, tr, fps)
        f = np.clip(ns.on_f, 0, max(0, len(tr) - 1))
        out["present"] = tr.present[f] if len(tr) else np.zeros((len(ns), 2), bool)
    return {k: v.astype(np.float32) if v.dtype != bool else v for k, v in out.items()}


def save_note_npz(path: Path, ns: NoteSet, probs: Dict[str, Optional[np.ndarray]],
                  rules: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    """One recording's per-note outputs, in playing order (read by hand_report)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"rows": ns.rows, "onset": np.array([n.onset for n in ns.notes]),
           "offset": np.array([n.offset for n in ns.notes]),
           "pitch": np.array([n.pitch for n in ns.notes], np.int64),
           "auto": ns.auto, "gt": ns.gt}
    for k, v in probs.items():
        if v is not None:
            arr[k] = np.asarray(v, np.float32)
    arr.update(rules)
    arr["meta"] = np.array(json.dumps({**ns.meta, **meta}))
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(path)


# ===================================================================== folds
def usable_recordings(cfg: Dict[str, Any], splits: Sequence[str]) -> List[Recording]:
    """Recordings of ``splits`` with automatic labels and landmarks (cache or JSON)."""
    excl = set(cfg["data"].get("exclude_records", []) or [])
    out = []
    for r in recordings_from_cfg(cfg):
        if r.split not in splits or r.record_time in excl:
            continue
        if not H.fingering_path(cfg, r.record_time).exists():
            continue
        cp = H.skeleton_cache_path(cfg, r.record_time)
        if not ((cp is not None and cp.exists()) or H.skeleton_json_path(cfg, r.record_time).exists()):
            continue
        out.append(r)
    return out


def make_folds(cfg: Dict[str, Any], scheme: str,
               recs: Optional[List[Recording]] = None) -> List[Tuple[str, List[str], List[str]]]:
    """[(fold name, training record_times, evaluated record_times)]."""
    hn = hn_cfg(cfg)
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if scheme == "official":
        tr_s = list(hn.get("official_train_splits", ["train", "ext-train"]))
        ev_s = list(hn.get("official_eval_splits", ["valid", "test"]))
        pool = recs if recs is not None else usable_recordings(cfg, tr_s + ev_s)
        return [("official", [r.record_time for r in pool if r.split in tr_s],
                 [r.record_time for r in pool if r.split in ev_s])]
    pool = recs if recs is not None else usable_recordings(
        cfg, list(hn.get("pool_splits", ["train", "ext-train", "valid", "test"])))
    gt = [r for r in pool if H.fingering_path(cfg, r.record_time, gt=True).exists()]
    if scheme == "loro":
        groups = [(r.record_time, {r.record_time}) for r in gt]
        ev = {r.record_time: [r.record_time] for r in gt}
    elif scheme == "lopo":
        names = sorted({r.performer for r in gt})
        groups = [(p, {r.record_time for r in pool if r.performer == p}) for p in names]
        ev = {p: [r.record_time for r in gt if r.performer == p] for p in names}
    else:
        names = sorted({session_of(cfg, r.record_time) for r in gt})
        groups = [(s, {r.record_time for r in pool if session_of(cfg, r.record_time) == s}) for s in names]
        ev = {s: [r.record_time for r in gt if session_of(cfg, r.record_time) == s] for s in names}
    return [(name.replace(" ", "_"), [r.record_time for r in pool if r.record_time not in out], ev[name])
            for name, out in groups]


# ===================================================================== train
class Log:
    def __init__(self, path: Optional[Path]):
        self.f = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()


def pick_device(cfg: Dict[str, Any]) -> str:
    torch = _torch()
    d = str((cfg.get("train") or {}).get("device", "auto"))
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def note_accuracy(sets: Sequence[NoteSet], probs: Sequence[np.ndarray], labels: str = "auto") -> float:
    ok = tot = 0
    for ns, p in zip(sets, probs):
        y = ns.auto if labels == "auto" else ns.gt
        m = y >= 0
        cls, _ = H.decide(ns.notes, np.log(np.clip(p, 1e-12, 1)))
        ok += int((cls[m] == y[m]).sum())
        tot += int(m.sum())
    return 100.0 * ok / max(1, tot)


def train_model(cfg: Dict[str, Any], train_sets: Sequence[NoteSet], seed: int, log: Log,
                monitor: Sequence[NoteSet] = ()):
    """Train a NoteHandModel for ``hand_notes.steps`` steps; returns (model, prior)."""
    torch = _torch()
    hn = hn_cfg(cfg)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(cfg)
    model = build_note_model(cfg).to(device)
    model.hn = hn
    W = int(hn.get("window_notes", 64))
    sets = [s for s in train_sets if len(s) >= W]
    if len(sets) < len(train_sets):
        log(f"  ({len(train_sets) - len(sets)} training recordings with < {W} notes left out)")
    if not sets:
        raise SystemExit("no training recordings")
    manual_frac = float(hn.get("manual_frac", 0.0))
    if manual_frac > 0:
        sets, n_man = with_manual_labels(sets)
        n_rec = sum(1 for s in sets if (s.gt >= 0).any())
        if n_man:
            log(f"  manual labels of the training recordings: {n_man} notes in {n_rec} recordings "
                f"({100 * manual_frac:.0f} % of the windows contain one)")
        else:
            log("  manual labels of the training recordings: none (no manually labelled recording "
                "in this fold's training set)")
            manual_frac = 0.0
    lab = np.array([float((s.y >= 0).sum()) for s in sets])
    weights = lab / lab.sum()
    prior = H.FingerPrior.fit([(s.notes, s.y) for s in sets], float(hn.get("chord_tol_s", 0.034)))
    a = dict(cfg.get("augment") or {})
    kb = cfg.get("keyboard") or {}
    aug = None
    if bool(hn.get("augment", True)):
        aug = {k: a.get(k, 0.0) for k in ("hscale", "hshift_px", "vshift_px", "rotate_deg")}
        aug.update(p=float(a.get("p", 0.8)) if a.get("enable", False) else 0.0,
                   strip_w=int(kb.get("warp_width", 1408)), strip_h=int(kb.get("warp_height", 112)),
                   corner_jitter=float(hn.get("corner_jitter", 0.0)),
                   hand_dropout=float((cfg.get("hands") or {}).get("skeleton_dropout", 0.1)))
    steps = int(hn.get("steps", 3000))
    B = int(hn.get("batch_windows", 16))
    lr = float(hn.get("lr", 1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(hn.get("weight_decay", 1e-4)))
    warm = max(1, int(float(hn.get("warmup_frac", 0.03)) * steps))

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        q = (s - warm) / max(1, steps - warm)
        return lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * q)))

    use_amp = bool(hn.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sup = str(hn.get("supervision", "finger"))
    wf, wn = float(hn.get("frame_loss_weight", 1.0)), float(hn.get("note_loss_weight", 1.0))
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  model: features={model.features} sequence={model.sequence} supervision={sup} "
        f"| {n_par:,} parameters | {len(sets)} training recordings, {int(lab.sum()):,} labelled notes "
        f"| {steps} steps x {B} windows x {W} notes | device {device}")
    t0 = time.time()
    run = {"loss": 0.0, "nl": 0.0, "fl": 0.0, "acc": 0.0, "n": 0}
    log_every = int(hn.get("log_every", 100))
    eval_every = int(hn.get("eval_every", 500))
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = to_device(sample_batch(sets, weights, B, W, model.margin, model.clip_len,
                                   model.features, aug, rng, manual_frac), device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(b["clips"], b["clip_of"], b["key"], b["n_read"], b["sym"])
        loss_n = class_nll(out["note_logits"], b["y"], sup)
        loss = wn * loss_n
        loss_f = None
        if out["frame_logits"] is not None and wf > 0:
            fl, mask = out["frame_logits"], out["frame_mask"] > 0
            yf = b["y"][..., None].expand(mask.shape).clone()
            yf[~mask] = H.IGNORE
            loss_f = class_nll(fl, yf, sup)
            loss = loss + wf * loss_f
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(hn.get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            m = b["y"] >= 0
            acc = (out["note_logits"].argmax(-1)[m] == b["y"][m]).float().mean().item() if m.any() else 0.0
        run["loss"] += loss.item(); run["nl"] += loss_n.item()
        run["fl"] += loss_f.item() if loss_f is not None else 0.0
        run["acc"] += acc; run["n"] += 1
        if (step + 1) % log_every == 0 or step + 1 == steps:
            k = max(1, run["n"])
            el = time.time() - t0
            log(f"  step {step + 1:5d}/{steps} lr {lr_at(step):.2e} loss {run['loss'] / k:.4f} "
                f"(note {run['nl'] / k:.4f} frame {run['fl'] / k:.4f}) batch note acc {100 * run['acc'] / k:5.1f} "
                f"| {el / 60:.1f} min, eta {(steps - step - 1) * el / (step + 1) / 60:.1f} min")
            run = {"loss": 0.0, "nl": 0.0, "fl": 0.0, "acc": 0.0, "n": 0}
        if monitor and ((step + 1) % eval_every == 0 or step + 1 == steps):
            pf, ps = zip(*[predict_noteset(model, s, device) for s in monitor])
            msg = f"  monitor (automatic labels of {len(monitor)} recordings): sequence {note_accuracy(monitor, ps):.2f}"
            if pf[0] is not None:
                msg += f", frame read-out {note_accuracy(monitor, pf):.2f}"
            log(msg + " % hand+finger")
    el = time.time() - t0
    log(f"  trained in {el / 60:.1f} min ({el / max(1, steps):.3f} s/step)")
    return model, prior


def save_model(path: Path, model, cfg: Dict[str, Any], prior: H.FingerPrior,
               info: Dict[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    torch.save({"model": model.state_dict(), "cfg": cfg, "prior": prior.to_dict(), "info": info}, tmp)
    tmp.replace(path)


def predict_and_save(model, cfg: Dict[str, Any], eval_sets: Sequence[NoteSet], out_dir: Path,
                     meta: Dict[str, Any], rule_cache: Dict[str, Dict[str, np.ndarray]], log: Log) -> None:
    device = next(model.parameters()).device
    for ns in eval_sets:
        rt = ns.meta["record_time"]
        if rt not in rule_cache:
            rule_cache[rt] = rule_scores(ns, cfg)
        pf, ps = predict_noteset(model, ns, device)
        save_note_npz(out_dir / f"{rt}.npz", ns, {"p_frame": pf, "p_seq": ps}, rule_cache[rt], meta)
        m = ns.gt >= 0
        if m.any():
            cls, _ = H.decide(ns.notes, np.log(np.clip(ps, 1e-12, 1)))
            msg = f"    {rt}: {int(m.sum())} manual notes, sequence {100 * (cls[m] == ns.gt[m]).mean():.1f}"
            if pf is not None:
                c2, _ = H.decide(ns.notes, np.log(np.clip(pf, 1e-12, 1)))
                msg += f", frame read-out {100 * (c2[m] == ns.gt[m]).mean():.1f}"
            roi, _ = H.decide(ns.notes, rule_cache[rt]["s_roi"]) if "s_roi" in rule_cache[rt] else (None, None)
            if roi is not None:
                msg += f", key-ROI rule {100 * (roi[m] == ns.gt[m]).mean():.1f}"
            log(msg + " % hand+finger")


def run_fold(cfg: Dict[str, Any], name: str, train_rts: Sequence[str], eval_rts: Sequence[str],
             seed: int, out: Path, pool: Dict[str, NoteSet], rule_cache: Dict[str, Dict[str, np.ndarray]],
             scheme: str, monitor_rts: Sequence[str] = ()) -> None:
    fd = out / f"{name}_s{seed}"
    if (fd / "DONE").exists():
        print(f"[{name} seed {seed}] done already ({fd}); skipped", flush=True)
        return
    fd.mkdir(parents=True, exist_ok=True)
    log = Log(fd / "train.log")
    tr = [pool[r] for r in train_rts if r in pool]
    ev = [pool[r] for r in eval_rts if r in pool]
    mon = [pool[r] for r in monitor_rts if r in pool]
    leak = set(train_rts) & set(eval_rts)
    if leak:
        raise SystemExit(f"fold {name}: evaluated recordings in training: {sorted(leak)}")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} fold {name} seed {seed}: train {len(tr)} recordings, "
        f"evaluate {len(ev)} ({', '.join(eval_rts)})")
    t0 = time.time()
    model, prior = train_model(cfg, tr, seed, log, mon)
    info = {"scheme": scheme, "fold": name, "seed": seed, "train": list(train_rts), "eval": list(eval_rts),
            "minutes": round((time.time() - t0) / 60, 2)}
    save_model(fd / "model.pt", model, cfg, prior, info)
    (fd / "prior.json").write_text(json.dumps(prior.to_dict()))
    (fd / "fold.json").write_text(json.dumps(info, indent=1))
    predict_and_save(model, cfg, ev, fd / "pred",
                     {"scheme": scheme, "fold": name, "seed": seed,
                      "features": str(hn_cfg(cfg).get("features", "keyrel"))}, rule_cache, log)
    (fd / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M"))
    log(f"=== fold {name} seed {seed} done in {(time.time() - t0) / 60:.1f} min -> {fd}")


def load_pool(cfg: Dict[str, Any], rts: Sequence[str], keep_track: Sequence[str] = ()) -> Dict[str, NoteSet]:
    by = {r.record_time: r for r in recordings_from_cfg(cfg)}
    keep = set(keep_track)
    pool = {}
    t0 = time.time()
    for i, rt in enumerate(sorted(set(rts))):
        try:
            pool[rt] = load_noteset(cfg, by[rt], keep_track=rt in keep)
        except (FileNotFoundError, KeyError) as e:
            print(f"  {rt}: [skipped] {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  loaded {i + 1}/{len(set(rts))} recordings ({time.time() - t0:.0f} s)", flush=True)
    return pool


def cmd_cv(args, cfg: Dict[str, Any], schemes: Sequence[str]) -> None:
    hn = hn_cfg(cfg)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    for scheme in schemes:
        folds = make_folds(cfg, scheme)
        if args.folds:
            want = {f.strip() for f in args.folds.split(",")}
            folds = [f for f in folds if f[0] in want]
        lim = int(hn.get("limit_recordings", 0))
        if lim:      # smoke tests only
            folds = [(n, sorted(tr)[:lim], ev) for n, tr, ev in folds]
        todo = [(f, s) for f in folds for s in seeds if not (out / f"{f[0]}_s{s}" / "DONE").exists()]
        print(f"{scheme}: {len(folds)} folds x {len(seeds)} seeds, {len(todo)} to run -> {out}", flush=True)
        if not todo:
            continue
        need = sorted({r for f, _ in todo for r in f[1] + f[2]})
        evals = sorted({r for f, _ in todo for r in f[2]})
        split_of = {x.record_time: x.split for x in recordings_from_cfg(cfg)}
        mon_splits = set(hn.get("monitor_splits", ["valid"]))
        mon = [r for r in need if scheme == "official" and split_of.get(r) in mon_splits]
        print(f"loading {len(need)} recordings ...", flush=True)
        pool = load_pool(cfg, need, keep_track=evals)
        rule_cache: Dict[str, Dict[str, np.ndarray]] = {}
        for (name, tr, ev), seed in todo:
            run_fold(cfg, name, tr, ev, seed, out, pool, rule_cache, scheme, mon)


# ===================================================================== robustness
PERTURBATIONS = ("corner", "noise", "swap", "drop")


def perturb_track(track: H.SkeletonTrack, corners: np.ndarray, kind: str, level: float,
                  rng: np.random.Generator) -> H.SkeletonTrack:
    """A test-time corruption of a recording's landmarks (all 21 per hand):
      corner  keyboard corners off by N(0, level x keyboard width), ONE draw per
              recording (a keyboard detector's error on a fixed camera);
              landmarks re-projected with the wrong corners
      noise   every landmark moved by N(0, level) white-key widths (x) and the
              same physical distance along the keys (y)
      swap    the two hands' tags exchanged on a random ``level`` share of frames
      drop    each hand missing on a random ``level`` share of frames"""
    pts = track.pts.copy()
    pres = track.present.copy()
    if kind == "corner":
        c = np.asarray(corners, np.float64)
        width = float(np.linalg.norm(c[1] - c[0]))
        moved = c + rng.normal(0.0, level * width, c.shape)
        m = H.strip_homography(moved) @ np.linalg.inv(H.strip_homography(c))
        xy = H.apply_homography(m, pts[..., :2])
        pts[..., :2] = np.where(pres[..., None, None], xy, 0.0)
    elif kind == "noise":
        sx, sy = level / H.N_WHITE, level / H.KEY_ASPECT
        pts[..., 0] += rng.normal(0, sx, pts[..., 0].shape).astype(np.float32)
        pts[..., 1] += rng.normal(0, sy, pts[..., 1].shape).astype(np.float32)
        pts[~pres] = 0.0
    elif kind == "swap":
        f = rng.random(len(pts)) < level
        pts[f] = pts[f][:, ::-1]
        pres[f] = pres[f][:, ::-1]
    elif kind == "drop":
        pres &= ~(rng.random(pres.shape) < level)
        pts[~pres] = 0.0
    else:
        raise ValueError(f"perturbation must be one of {PERTURBATIONS}")
    return H.SkeletonTrack(pts.astype(np.float32), pres)


def cmd_robust(args, cfg: Dict[str, Any]) -> None:
    """Re-predict the held-out recordings of trained folds under perturbations
    (``--tests corner:0.01,swap:0.05,...``); outputs in <fold>/pred_<kind><level>/."""
    torch = _torch()
    import zlib
    from dataclasses import replace

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    tests = []
    for t in args.tests.split(","):
        k, _, v = t.strip().partition(":")
        if k not in PERTURBATIONS or not v:
            raise SystemExit(f"bad test {t!r}: want kind:level, kind in {PERTURBATIONS}")
        tests.append((k, float(v)))
    seeds = {int(x) for x in str(args.seeds).split(",") if x.strip()}
    folds = sorted(d for d in run_dir.iterdir() if (d / "DONE").exists() and (d / "fold.json").exists())
    pool: Dict[str, NoteSet] = {}
    by = {r.record_time: r for r in recordings_from_cfg(cfg)}
    for fd in folds:
        info = json.loads((fd / "fold.json").read_text())
        if int(info["seed"]) not in seeds:
            continue
        model, mcfg, _ = load_note_model(fd / "model.pt", device)
        for rt in info["eval"]:
            if rt not in pool:
                pool[rt] = load_noteset(cfg, by[rt], keep_track=True)
            ns = pool[rt]
            for kind, level in tests:
                out = fd / f"pred_{kind}{level:g}" / f"{rt}.npz"
                if out.exists():
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{rt}|{kind}|{level}".encode()))
                tr = perturb_track(ns.track, ns.corners, kind, level, rng)
                ns2 = replace(ns, skel=tr.model_input(np.arange(len(tr))), track=tr)
                pf, ps = predict_noteset(model, ns2, device)
                save_note_npz(out, ns2, {"p_frame": pf, "p_seq": ps}, rule_scores(ns2, mcfg),
                              {"scheme": info["scheme"], "fold": info["fold"], "seed": info["seed"],
                               "perturbation": f"{kind}:{level:g}"})
        print(f"  {fd.name}: {len(info['eval'])} recordings x {len(tests)} perturbations", flush=True)
    names = " ".join(f"pred_{k}{v:g}" for k, v in tests)
    print(f"done. Report one with: python -m pianovam_vision.hand_report --run x={run_dir} --pred_dir <one of: {names}>")


# ===================================================================== end to end
def e2e_counts(ref: Sequence[Note], lab: np.ndarray, est: Sequence[Note], cls: np.ndarray,
               tol: float) -> Dict[str, int]:
    """Note / +hand / +finger matches of transcribed notes with hand decisions
    against labelled reference notes (onset within ``tol``, pitchwise, as
    evaluate_hands). Unlabelled reference notes (and the estimates matched to
    them) are left out. Error split of the labelled reference notes: missed,
    wrong hand, right hand wrong finger, correct; and extra estimates."""
    from .metrics import _pitchwise_matching

    pairs = _pitchwise_matching(list(ref), list(est), tol, None, 0.05)
    c = {"ref": int((lab >= 0).sum()), "est": len(est), "match": 0, "hand": 0, "finger": 0}
    matched_est = set()
    for ri, ei in pairs:
        matched_est.add(ei)
        y = int(lab[ri])
        if y < 0:
            c["est"] -= 1
            continue
        p = int(cls[ei])
        c["match"] += 1
        c["hand"] += int(p >= 0 and p // 5 == y // 5)
        c["finger"] += int(p >= 0 and p == y)
    c["missed"] = c["ref"] - c["match"]
    c["wrong_hand"] = c["match"] - c["hand"]
    c["wrong_finger"] = c["hand"] - c["finger"]
    c["extra"] = c["est"] - c["match"]
    return c


def f1(tp: int, n_est: int, n_ref: int) -> float:
    p, r = tp / max(1, n_est), tp / max(1, n_ref)
    return 100.0 * (2 * p * r / (p + r) if p + r > 0 else 0.0)


def cmd_e2e(args, cfg_cli: Dict[str, Any]) -> None:
    import csv

    from .evaluate_hands import load_references
    from .score_midi import load_midi_notes

    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, prior = load_note_model(args.model, device)
    cfg = dict(cfg_cli)
    for k in ("model", "hands", "hand_notes", "labels"):
        cfg[k] = mcfg[k]
    preds = {}
    for p in args.pred:
        name, _, d = p.partition("=")
        if not d:
            raise SystemExit(f"--pred wants NAME=DIR, got {p!r}")
        preds[name] = Path(d)
    splits = [s.strip() for s in args.split.split(",")]
    recs = [r for r in usable_recordings(cfg, splits)]
    if args.records:
        want = {x.strip() for x in args.records.split(",")}
        recs = [r for r in recs if r.record_time in want]
    out = Path(args.out)
    tols = (0.05, 0.10)
    methods = ["model", "roi", "ayf"]
    rows, tot = [], {}
    print(f"end-to-end hands on {','.join(splits)}: {len(recs)} recordings | notes from {', '.join(preds)} "
          f"| model {args.model}", flush=True)
    fps = float(cfg["labels"]["fps"])
    for rec in recs:
        ref, auto, gt = load_references(cfg, rec)
        track = H.load_skeleton(cfg, rec)
        skel = track.model_input(np.arange(len(track)))
        label_sets = [("automatic", auto, ref)]
        if gt is not None and len(gt):
            n_gt = len(gt)
            t_end = (ref[n_gt - 1].onset + ref[n_gt].onset) / 2 if n_gt < len(ref) else float("inf")
            label_sets.append(("manual", gt, ref[:n_gt], t_end))
        for src, d in preds.items():
            f = d / f"{rec.record_time}.mid"
            if not f.exists():
                print(f"  {rec.record_time}: no {f}; {src} skipped here", flush=True)
                continue
            est = load_midi_notes(f)
            ns = make_noteset(est, cfg, skel, rec.corners, track=track,
                              meta={"record_time": rec.record_time})
            _, ps = predict_noteset(model, ns, device)
            rs = rule_scores(ns, cfg)
            dec = {"model": H.decide(ns.notes, np.log(np.clip(ps, 1e-12, 1)))[0],
                   "roi": H.decide(ns.notes, rs["s_roi"])[0],
                   "ayf": H.decide(ns.notes, rs["s_ayf"], joint=True)[0]}
            conf = H.decide(ns.notes, np.log(np.clip(ps, 1e-12, 1)))[1]
            H.write_hand_tsv(out / src / f"{rec.record_time}_hands.tsv", ns.notes, dec["model"], conf)
            H.write_hand_midi(out / src / f"{rec.record_time}_hands.mid", ns.notes, dec["model"])
            for ls in label_sets:
                lname, lab, refs = ls[0], ls[1], ls[2]
                if lname == "manual":
                    sel = [i for i, n in enumerate(ns.notes) if n.onset < ls[3]]
                else:
                    sel = list(range(len(ns)))
                est_sub = [ns.notes[i] for i in sel]
                for m in methods:
                    for tol in tols:
                        c = e2e_counts(refs, np.asarray(lab), est_sub, dec[m][sel], tol)
                        ms = int(round(tol * 1000))
                        r = {"record": rec.record_time, "labels": lname, "notes": src, "method": m,
                             "tol_ms": ms, **c,
                             "note_f1": round(f1(c["match"], c["est"], c["ref"]), 2),
                             "hand_f1": round(f1(c["hand"], c["est"], c["ref"]), 2),
                             "finger_f1": round(f1(c["finger"], c["est"], c["ref"]), 2)}
                        rows.append(r)
                        t = tot.setdefault((lname, src, m, ms), {k: 0 for k in c})
                        for k, v in c.items():
                            t[k] += v
        print(f"  {rec.record_time} [{rec.split}]: {len(ref)} notes"
              + (f", {len(gt)} manual" if gt is not None else ""), flush=True)

    for lname in ("automatic", "manual"):
        keys = sorted({k for k in tot if k[0] == lname})
        if not keys:
            continue
        print(f"\n=== end-to-end, {lname} labels: F1 % at 50 / 100 ms -- mean over recordings (pooled) ===")
        print(f"  {'notes from':12s} {'hands by':8s} {'Note':>21s} {'+Hand':>21s} {'+Hand+Finger':>21s}")
        for src in preds:
            for m in methods:
                cells = []
                for key in ("note_f1", "hand_f1", "finger_f1"):
                    parts = []
                    for ms in (50, 100):
                        per = [r[key] for r in rows if (r["labels"], r["notes"], r["method"], r["tol_ms"]) ==
                               (lname, src, m, ms)]
                        t = tot.get((lname, src, m, ms))
                        if not per or t is None:
                            parts.append("  -  ")
                            continue
                        tp = {"note_f1": t["match"], "hand_f1": t["hand"], "finger_f1": t["finger"]}[key]
                        parts.append(f"{np.mean(per):4.1f} ({f1(tp, t['est'], t['ref']):4.1f})")
                    cells.append(" / ".join(parts))
                print(f"  {src:12s} {m:8s} " + "  ".join(f"{c:>21s}" for c in cells))
        print("  error split of the labelled reference notes at 100 ms (pooled, %): missed / wrong hand / "
              "right hand wrong finger / correct; extra estimates per reference note")
        for src in preds:
            for m in methods:
                t = tot.get((lname, src, m, 100))
                if not t:
                    continue
                R = max(1, t["ref"])
                print(f"    {src:12s} {m:8s} {100 * t['missed'] / R:5.1f} / {100 * t['wrong_hand'] / R:4.1f} / "
                      f"{100 * t['wrong_finger'] / R:4.1f} / {100 * t['finger'] / R:5.1f}; extra {100 * t['extra'] / R:4.1f}")
    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nrows -> {args.csv}")
    print(f"hand-separated MIDI + TSV per recording -> {out}/<notes from>/")


# ===================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "cv", "folds", "e2e", "robust"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("overrides", nargs="*")
        if name in ("train", "cv"):
            p.add_argument("--out", required=True)
            p.add_argument("--seeds" if name == "cv" else "--seed", default="1")
            p.add_argument("--folds", default="", help="only these fold names (comma separated)")
        if name == "train":
            p.add_argument("--fold", default="official", help="official | loro:<rec> | lopo:<pianist> | session:<name>")
        if name in ("cv", "folds"):
            p.add_argument("--scheme", required=True, help=f"comma separated, of {','.join(SCHEMES)}")
        if name == "e2e":
            p.add_argument("--model", required=True)
            p.add_argument("--pred", action="append", default=[], help="NAME=DIR of <record_time>.mid")
            p.add_argument("--split", default="test")
            p.add_argument("--out", required=True)
            p.add_argument("--csv", default=None)
            p.add_argument("--records", default="", help="only these recordings (comma separated)")
        if name == "robust":
            p.add_argument("--run_dir", required=True, help="a cv output directory")
            p.add_argument("--seeds", default="1")
            p.add_argument("--tests", default="corner:0.005,corner:0.01,corner:0.02,noise:0.1,noise:0.25,"
                                              "swap:0.05,swap:0.2,drop:0.2,drop:0.5")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.cmd == "folds":
        for scheme in args.scheme.split(","):
            folds = make_folds(cfg, scheme.strip())
            print(f"{scheme}: {len(folds)} folds")
            for name, tr, ev in folds:
                print(f"  {name:22s} train {len(tr):3d} recordings | evaluate {', '.join(ev)}")
        return
    if args.cmd == "e2e":
        cmd_e2e(args, cfg)
        return
    if args.cmd == "robust":
        cmd_robust(args, cfg)
        return
    if args.cmd == "train":
        scheme, _, which = args.fold.partition(":")
        args.seeds = args.seed
        args.folds = which if which else ""
        cmd_cv(args, cfg, [scheme])
        return
    cmd_cv(args, cfg, [s.strip() for s in args.scheme.split(",")])


if __name__ == "__main__":
    main(sys.argv[1:])
