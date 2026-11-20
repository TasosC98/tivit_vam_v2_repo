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

* ``r3`` -- the raw R3 release (archive.org ``r3dataset_raw``, Zivanovic et al.,
  IJCAI 2025): one folder per subset/split (``r3s_train``, ``r3s_test``,
  ``train`` and ``test``; the last two are R3x), each holding chunks
  ``<session>_full_split<N>.mp4`` + ``<session>_split<N>.mid``, plus one
  keyboard box per session in ``r3s_bounding_boxes.json`` /
  ``r3x_bounding_boxes.json``. One chunk is one recording (``record_time`` = the
  MIDI stem, the name V2N's released predictions use). See ``load_recordings_r3``
  for the box geometry; valid is held out from train by recording DATE (the
  release keeps train and test apart by date).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
    # R3 only: subset ("r3s" / "r3x", reported separately) and session name.
    subset: str = ""
    session: str = ""

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


# --------------------------------------------------------------------------- R3
# folder -> (subset, split) of the archive.org release
R3_SPLIT_DIRS: Dict[str, Sequence[str]] = {
    "r3s_train": ("r3s", "train"), "r3s_test": ("r3s", "test"),
    "train": ("r3x", "train"), "test": ("r3x", "test"),
}
R3_BOX_FILES: Dict[str, str] = {"r3s": "r3s_bounding_boxes.json",
                                "r3x": "r3x_bounding_boxes.json"}
# Order of the straightened box's image corners [TL, TR, BR, BL] that gives the
# keyboard corners [LT, RT, RB, LB] (LT = A0 end, key backs; the warp puts it at
# the strip's top left).
R3_ORIENTATIONS: Dict[str, Sequence[int]] = {
    # R3's camera hangs over the piano and faces the pianist: the picture is the
    # usual view turned by 180 degrees (A0 at the image's RIGHT, black keys BELOW
    # the white ones). Checked on all 143 sessions and with the MIDI notes drawn
    # on real frames (2026-11-17).
    "rot180": (2, 3, 0, 1),
    "normal": (0, 1, 2, 3),   # A0 at the left, key backs at the top (PianoVAM view)
}


def r3_box_corners(box: Mapping[str, Any], frame_size: Sequence[float] = (1920, 1080),
                   orientation: str = "rot180") -> np.ndarray:
    """Keyboard corners [LT, RT, RB, LB] in raw-frame pixels from one R3 box.

    The release gives ``rot_angle`` (degrees) and an axis-aligned box ``x1 y1 x2
    y2`` in the frame straightened by turning it about its centre CLOCKWISE on
    screen by ``rot_angle`` (OpenCV: ``getRotationMatrix2D(centre, -rot_angle)``;
    the opposite sign misses the keyboard by up to ~30 px on the 3-degree
    sessions). The box corners are turned back into the raw frame, so the
    perspective warp straightens the keyboard itself.
    """
    if orientation not in R3_ORIENTATIONS:
        raise ValueError(f"R3 orientation must be one of {sorted(R3_ORIENTATIONS)}, "
                         f"got {orientation!r}")
    w, h = float(frame_size[0]), float(frame_size[1])
    a = math.radians(float(box["rot_angle"]))
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    straight = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    c = np.array([w / 2.0, h / 2.0])
    # straight = A @ (raw - c) + c, with A the on-screen clockwise turn by rot_angle
    A = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    raw = (straight - c) @ A + c          # row vectors: (A^T @ (p - c)^T)^T == (p - c) @ A
    return raw[list(R3_ORIENTATIONS[orientation])].astype(np.float32)


def r3_date(session: str) -> str:
    """Recording date of an R3 session name (``ls_2024-02-06_a02`` -> ``2024-02-06``)."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", session)
    return m.group(0) if m else session


def load_recordings_r3(
    root: str | Path,
    *,
    split_dirs: Mapping[str, Sequence[str]] | None = None,
    box_files: Mapping[str, str] | None = None,
    subsets: Iterable[str] | None = None,
    video_pattern: str = "{session}_full_split{chunk}.mp4",
    box_key_pattern: str = "{session}_full",
    frame_size: Sequence[float] = (1920, 1080),
    orientation: str = "rot180",
    valid_frac: float = 0.1,
    valid_seed: int = 1234,
    valid_avoid_sessions: Iterable[str] = (),
    corners: Mapping[str, Any] | None = None,
) -> List[Recording]:
    """Index the raw R3 release (see the module docstring) into Recordings.

    ``corners``: annotated keyboard corners per session ({session: 4 points
    [A0 back, C8 back, C8 front, A0 front] in native frame pixels}, see
    ``load_r3_corners``), used instead of the official boxes. Then EVERY session
    must be in it: a session left out raises (no silent mix of the two).

    Every ``<session>_split<N>.mid`` in a split folder is one recording; its
    video is ``video_pattern`` in the same folder. ``subsets`` keeps only those
    subsets (default: all). Valid: per subset, whole recording DATES are moved
    from train to valid, in a fixed random order (``valid_seed``), until about
    ``valid_frac`` of that subset's train chunks are valid (a date that would
    overshoot by more than is still missing is skipped); dates with a session in
    ``valid_avoid_sessions`` (e.g. known video/MIDI sync faults) are never
    picked. The choice for one subset does not depend on the other.
    """
    root = Path(root)
    split_dirs = dict(split_dirs or R3_SPLIT_DIRS)
    box_files = dict(box_files or R3_BOX_FILES)
    keep = set(subsets) if subsets else None
    boxes: Dict[str, Dict[str, Any]] = {}
    recs: List[Recording] = []
    missing: set = set()
    for dname, (sub, split) in split_dirs.items():
        if keep is not None and sub not in keep:
            continue
        folder = root / dname
        if not folder.is_dir():
            found = sorted(p.name for p in root.iterdir()) if root.is_dir() else "root missing"
            raise FileNotFoundError(
                f"R3: no folder {folder} (data.r3_split_dirs maps folder -> [subset, split]); "
                f"{root} contains: {found}")
        if sub not in boxes:
            bpath = root / box_files[sub]
            if not bpath.exists():
                raise FileNotFoundError(f"R3: keyboard boxes not found: {bpath}")
            with open(bpath, "r", encoding="utf-8") as f:
                boxes[sub] = json.load(f)
        for mid in sorted(folder.glob("*.mid")):
            m = re.fullmatch(r"(.+)_split(\d+)", mid.stem)
            if not m:
                continue
            session, chunk = m.group(1), int(m.group(2))
            key = box_key_pattern.format(session=session)
            if key not in boxes[sub]:
                raise KeyError(f"R3: no keyboard box '{key}' in {box_files[sub]} for {mid.name}")
            if corners is not None:
                if session not in corners:
                    missing.add(session)
                    continue
                c = np.asarray(corners[session], dtype=np.float32)
            else:
                c = r3_box_corners(boxes[sub][key], frame_size, orientation)
            recs.append(Recording(
                record_time=mid.stem,
                split=split,
                corners=c,
                video_filename=f"{dname}/{video_pattern.format(session=session, chunk=chunk)}",
                label_filename=f"{dname}/{mid.name}",
                subset=sub,
                session=session,
            ))

    if missing:
        raise KeyError(f"R3: {len(missing)} session(s) have no annotated corners "
                       f"(data.r3_corners): {sorted(missing)}")

    if valid_frac and valid_frac > 0:
        avoid = set(valid_avoid_sessions or ())
        for sub in sorted({r.subset for r in recs}):
            train = [r for r in recs if r.subset == sub and r.split == "train"]
            by_date: Dict[str, List[Recording]] = defaultdict(list)
            for r in train:
                by_date[r3_date(r.session)].append(r)
            dates = sorted(d for d, rs in by_date.items()
                           if not any(r.session in avoid for r in rs))
            random.Random(f"{valid_seed}-{sub}").shuffle(dates)
            target, picked, n = valid_frac * len(train), [], 0
            for d in dates:
                if n >= target:
                    break
                k = len(by_date[d])
                # a date that would overshoot by more than is still missing is
                # skipped (a smaller one later in the order fits better)
                if n + k - target > target - n:
                    continue
                picked.append(d)
                n += k
            if not picked and dates:
                picked = [min(dates, key=lambda d: len(by_date[d]))]
            for d in picked:
                for r in by_date[d]:
                    r.split = "valid"
    return recs


def load_r3_corners(path: str | Path) -> Dict[str, np.ndarray]:
    """Annotated R3 keyboard corners (scripts/corner_tool.html -> JSON).

    ``{"format": "r3_corners_v1", "corners": {session: [[x, y] x 4]}}``, points
    [A0 back, C8 back, C8 front, A0 front] in native frame pixels. A relative
    path is looked up from the working directory, then from vision_transcription/.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parents[1] / p
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "r3_corners_v1":
        raise ValueError(f"{p}: not an r3_corners_v1 file")
    out = {}
    for session, pts in data["corners"].items():
        a = np.asarray(pts, dtype=np.float32)
        if a.shape != (4, 2) or not np.isfinite(a).all():
            raise ValueError(f"{p}: session {session}: need 4 [x, y] points, got {pts}")
        out[session] = a
    return out


def recordings_from_cfg(cfg: Dict[str, Any]) -> List[Recording]:
    """Load recordings for the active dataset, dispatching on ``data.format``."""
    d = cfg["data"]
    root = Path(d["root"])
    fmt = d.get("format", "pianovam")
    if fmt == "r3":
        return load_recordings_r3(
            root,
            split_dirs=d.get("r3_split_dirs"),
            box_files=d.get("r3_box_files"),
            subsets=d.get("subsets"),
            video_pattern=d.get("video_pattern", "{session}_full_split{chunk}.mp4"),
            box_key_pattern=d.get("box_key_pattern", "{session}_full"),
            frame_size=d.get("frame_size", (1920, 1080)),
            orientation=d.get("orientation", "rot180"),
            valid_frac=d.get("valid_frac", 0.1),
            valid_seed=d.get("valid_seed", 1234),
            valid_avoid_sessions=d.get("valid_avoid_sessions") or (),
            corners=load_r3_corners(d["r3_corners"]) if d.get("r3_corners") else None,
        )
    path = root / d["metadata"]
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
