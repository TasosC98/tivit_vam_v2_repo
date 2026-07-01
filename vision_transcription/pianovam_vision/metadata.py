"""Parse metadata_v2.json into per-recording records.

metadata_v2.json maps an integer string index -> a dict with, among others:
  record_time : str  e.g. "2024-02-14_19-10-09" (== file stem of mp4/tsv/mid)
  split       : str  one of train/valid/test/ext-train/special(blurry)/special(4hands)
  Point_LT, Point_RT, Point_RB, Point_LB : "x, y" pixel coords of the
              keyboard quadrilateral corners (top-left, top-right,
              bottom-right, bottom-left) in the 1920x1080 frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Recording:
    record_time: str
    split: str
    corners: np.ndarray          # (4, 2) float32 in order [LT, RT, RB, LB]
    composer: str = ""
    piece: str = ""
    performer: str = ""

    def video_path(self, root: Path, video_dir: str, ext: str) -> Path:
        return root / video_dir / f"{self.record_time}{ext}"

    def tsv_path(self, root: Path, tsv_dir: str) -> Path:
        return root / tsv_dir / f"{self.record_time}.tsv"

    def midi_path(self, root: Path, midi_dir: str) -> Path:
        return root / midi_dir / f"{self.record_time}.mid"


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


def filter_by_split(recs: Sequence[Recording], splits: Sequence[str]) -> List[Recording]:
    wanted = set(splits)
    return [r for r in recs if r.split in wanted]


def index_by_record_time(recs: Sequence[Recording]) -> Dict[str, Recording]:
    return {r.record_time: r for r in recs}
