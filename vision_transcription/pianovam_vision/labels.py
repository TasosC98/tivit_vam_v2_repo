"""TSV label parsing and per-frame target-roll construction.

TSV columns (header line starts with '#'):
    onset  key_offset  frame_offset  note  velocity   (all tab separated)

For VISUAL transcription we use ``key_offset`` (the moment the finger physically
leaves the key) as the note offset, because the pedal-extended ``frame_offset``
is not observable from the keyboard image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import N_KEYS, PITCH_MAX, PITCH_MIN


@dataclass
class Note:
    onset: float          # seconds
    offset: float         # seconds (key_offset or frame_offset per config)
    pitch: int            # MIDI note number
    velocity: int


def read_tsv(
    path: str | Path, offset_field: str = "key_offset"
) -> List[Note]:
    """Read a PianoVAM TSV into a list of Note (sorted by onset)."""
    col = {"onset": 0, "key_offset": 1, "frame_offset": 2, "note": 3, "velocity": 4}
    off_idx = col[offset_field]
    notes: List[Note] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            onset = float(parts[col["onset"]])
            offset = float(parts[off_idx])
            pitch = int(float(parts[col["note"]]))
            vel = int(float(parts[col["velocity"]]))
            if offset < onset:
                offset = onset
            notes.append(Note(onset, offset, pitch, vel))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def read_midi(path: str | Path) -> List[Note]:
    """Read a MIDI file (PianoYT ground truth) into a list of Note.

    PianoYT labels come as MIDI (extracted by Onsets-and-Frames from the audio),
    so there is only note-on/note-off -- the offset is the MIDI note-off time.
    NOTE: these offsets include sustain-pedal tails, which a camera cannot see,
    so onset+pitch is the fair metric for a visual model (offset F1 will be low).
    """
    import pretty_midi  # imported lazily so the package imports without pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes: List[Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            offset = n.end if n.end >= n.start else n.start
            notes.append(Note(float(n.start), float(offset), int(n.pitch), int(n.velocity)))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def reference_path(rec, cfg: Dict[str, Any]) -> Path:
    """Path to a recording's ground-truth label file, per ``data.format``."""
    d = cfg["data"]
    root = Path(d["root"])
    if d.get("format", "pianovam") == "pianoyt":
        return rec.label_path(root, d["midi_dir"])
    return rec.tsv_path(root, d["tsv_dir"])


_OFFSETS_CACHE: Dict[str, Dict[str, float]] = {}


def load_label_offsets(path: str | Path) -> Dict[str, float]:
    """Per-recording label time shifts (seconds), JSON ``{record_id: seconds}``.

    The shift is ADDED to every label time (onset and offset), i.e. it moves the
    labels onto the video timeline. Produced by ``sync_check --write_offsets``.
    """
    key = str(path)
    if key not in _OFFSETS_CACHE:
        import json
        with open(path, "r", encoding="utf-8") as f:
            _OFFSETS_CACHE[key] = {str(k): float(v) for k, v in json.load(f).items()}
    return _OFFSETS_CACHE[key]


def shift_notes(notes: List[Note], seconds: float) -> List[Note]:
    """Shift notes in time, dropping any that would start before t=0."""
    if not seconds:
        return notes
    out = [Note(n.onset + seconds, n.offset + seconds, n.pitch, n.velocity)
           for n in notes]
    return [n for n in out if n.onset >= 0.0]


def read_reference(rec, cfg: Dict[str, Any]) -> List[Note]:
    """Read a recording's ground-truth notes (TSV for PianoVAM, MIDI for PianoYT).

    If ``data.label_offsets`` names a JSON of per-recording time shifts (e.g.
    audio-derived PianoYT labels that are out of sync with the picture), the
    recording's shift is applied here, so training, evaluation and the preview
    tools all see the same corrected labels.
    """
    path = reference_path(rec, cfg)
    if cfg["data"].get("format", "pianovam") == "pianoyt":
        notes = read_midi(path)
    else:
        notes = read_tsv(path, cfg["labels"]["offset_field"])
    offsets_path = cfg["data"].get("label_offsets")
    if offsets_path:
        notes = shift_notes(notes, load_label_offsets(offsets_path).get(rec.record_time, 0.0))
    return notes


def build_target_rolls(
    notes: List[Note],
    num_frames: int,
    fps: float,
    onset_window_frames: int = 2,
    min_note_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct frame / onset / velocity target rolls.

    Returns
    -------
    frame_roll : (num_frames, 88) float32 in {0,1}   -- key held down
    onset_roll : (num_frames, 88) float32 in {0,1}   -- onset region
    velocity   : (num_frames, 88) float32 in [0,1]   -- velocity/127 at onset
    """
    frame_roll = np.zeros((num_frames, N_KEYS), dtype=np.float32)
    onset_roll = np.zeros((num_frames, N_KEYS), dtype=np.float32)
    velocity = np.zeros((num_frames, N_KEYS), dtype=np.float32)

    for n in notes:
        if not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        k = n.pitch - PITCH_MIN
        on_f = int(round(n.onset * fps))
        if on_f >= num_frames:
            # Starts after the covered frames (capped video, or labels longer
            # than the video). Clamping it would stack a fake onset of every such
            # note onto the last frame.
            continue
        off_f = int(round(n.offset * fps))
        if off_f < on_f + min_note_frames:
            off_f = on_f + min_note_frames
        on_f = max(0, min(on_f, num_frames - 1))
        off_f = max(on_f + 1, min(off_f, num_frames))

        frame_roll[on_f:off_f, k] = 1.0
        on_end = min(on_f + max(1, onset_window_frames), num_frames)
        onset_roll[on_f:on_end, k] = 1.0
        velocity[on_f:on_end, k] = n.velocity / 127.0

    return frame_roll, onset_roll, velocity
