"""Parse dataset metadata into per-recording ``Recording`` objects.

Two dataset layouts are supported, selected by ``data.format`` in the config:

* ``pianovam`` (default) -- ``metadata_v2.json`` maps an integer string index
  to a dict with, among others:
    record_time : str  e.g. "2024-02-14_19-10-09" (== file stem of mp4/tsv/mid)
    split       : str  train/valid/test/ext-train/special(blurry)/special(4hands)
    Point_LT, Point_RT, Point_RB, Point_LB : "x, y" pixel coords of the
                keyboard quadrilateral corners (top-left, top-right,
                bottom-right, bottom-left) in the 1920x1080 frame.

* ``pianoyt`` -- ``pianoyt.csv`` with one row per YouTube video:
    videoID, url, split(1=train/3=test), min_y, max_y, min_x, max_x
  The keyboard region is an axis-aligned crop box (not a perspective quad), and
  the labels are MIDI files (``audio_<id>.0.midi``), not TSVs. PianoYT has no
  validation split, so a deterministic fraction of train is held out as valid
  (see ``data.valid_frac``).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class Recording:
    record_time: str
    split: str
    corners: np.ndarray          # (4, 2) float32 in order [LT, RT, RB, LB]
    composer: str = ""
    piece: str = ""
    performer: str = ""
    # Explicit on-disk filenames (used by PianoYT, whose video/label names do not
    # equal ``record_time``). When empty, fall back to the PianoVAM convention.
    video_filename: str = ""
    label_filename: str = ""

    def video_path(self, root: Path, video_dir: str, ext: str) -> Path:
        name = self.video_filename or f"{self.record_time}{ext}"
        return root / video_dir / name

    def tsv_path(self, root: Path, tsv_dir: str) -> Path:
        return root / tsv_dir / f"{self.record_time}.tsv"

    def midi_path(self, root: Path, midi_dir: str) -> Path:
        return root / midi_dir / f"{self.record_time}.mid"

    def label_path(self, root: Path, label_dir: str) -> Path:
        """Path to the ground-truth label file (TSV for PianoVAM, MIDI for PianoYT)."""
        name = self.label_filename or f"{self.record_time}.tsv"
        return root / label_dir / name


def _parse_point(s: str) -> List[float]:
    x, y = s.split(",")
    return [float(x.strip()), float(y.strip())]


def load_recordings(metadata_path: str | Path) -> List[Recording]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    recs: List[Recording] = []
    for _, e in sorted(raw.items(), key=lambda kv: int(kv[0])):
        corners = np.array(
            [
                _parse_point(e["Point_LT"]),
                _parse_point(e["Point_RT"]),
                _parse_point(e["Point_RB"]),
                _parse_point(e["Point_LB"]),
            ],
            dtype=np.float32,
        )
        recs.append(
            Recording(
                record_time=e["record_time"],
                split=e["split"],
                corners=corners,
                composer=e.get("composer", "") or "",
                piece=e.get("piece", "") or "",
                performer=e.get("P1_name", "") or "",
            )
        )
    return recs


def load_recordings_pianoyt(
    csv_path: str | Path,
    *,
    video_pattern: str = "video_{id}.mp4",
    midi_pattern: str = "audio_{id}.0.midi",
    valid_frac: float = 0.1,
    valid_seed: int = 1234,
) -> List[Recording]:
    """Load pianoyt.csv rows into Recordings.

    CSV columns (no header): videoID, url, split(1=train/3=test),
    min_y, max_y, min_x, max_x. The crop box becomes the 4 keyboard corners
    (LT, RT, RB, LB); the box is axis-aligned so the "perspective" warp is just a
    crop + resize. A deterministic ``valid_frac`` slice of train is relabelled
    ``valid`` (PianoYT ships no validation split).

    ``video_pattern`` / ``midi_pattern`` are ``str.format`` templates taking
    ``{id}`` (e.g. ``"video_{id}.0.mp4"`` -> ``video_100.0.mp4``).
    """
    recs: List[Recording] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            vid = parts[0]
            split_code = int(parts[2])
            min_y, max_y, min_x, max_x = (float(parts[3]), float(parts[4]),
                                          float(parts[5]), float(parts[6]))
            corners = np.array(
                [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
                dtype=np.float32,
            )
            split = {1: "train", 3: "test"}.get(split_code, "other")
            recs.append(
                Recording(
                    record_time=vid,
                    split=split,
                    corners=corners,
                    video_filename=video_pattern.format(id=vid),
                    label_filename=midi_pattern.format(id=vid),
                )
            )

    if valid_frac and valid_frac > 0:
        train_ids = sorted((r.record_time for r in recs if r.split == "train"),
                           key=lambda s: int(s))
        k = max(1, round(len(train_ids) * valid_frac))
        valid_ids = set(random.Random(valid_seed).sample(train_ids, k))
        for r in recs:
            if r.record_time in valid_ids:
                r.split = "valid"
    return recs


def recordings_from_cfg(cfg: Dict[str, Any]) -> List[Recording]:
    """Load recordings for the active dataset, dispatching on ``data.format``."""
    d = cfg["data"]
    root = Path(d["root"])
    path = root / d["metadata"]
    fmt = d.get("format", "pianovam")
    if fmt == "pianoyt":
        return load_recordings_pianoyt(
            path,
            video_pattern=d.get("video_pattern", "video_{id}.mp4"),
            midi_pattern=d.get("midi_pattern", "audio_{id}.0.midi"),
            valid_frac=d.get("valid_frac", 0.1),
            valid_seed=d.get("valid_seed", 1234),
        )
    return load_recordings(path)


def filter_by_split(recs: Sequence[Recording], splits: Sequence[str]) -> List[Recording]:
    wanted = set(splits)
    return [r for r in recs if r.split in wanted]


def index_by_record_time(recs: Sequence[Recording]) -> Dict[str, Recording]:
    return {r.record_time: r for r in recs}
