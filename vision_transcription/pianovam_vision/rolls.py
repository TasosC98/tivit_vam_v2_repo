"""Save and reuse the model's probability rolls.

Running the model is the slow part of calibrate/evaluate; decoding is cheap. With
``--rolls_dir`` the onset/frame probabilities of every recording are stored once
(float32 .npz), and any later run with the same checkpoint and data decodes from
them without the GPU -- e.g. to try other thresholds or timing settings.

A ``meta.json`` records what the rolls were made from (checkpoint file, data,
frame rate, warp). If it does not match, the stored rolls are discarded and
recomputed, so a stale cache is never used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def rolls_meta(checkpoint: str, cfg: Dict[str, Any], split: str, max_frames: int) -> Dict[str, Any]:
    ck = Path(checkpoint)
    st = ck.stat()
    d, kb = cfg["data"], cfg["keyboard"]
    return {
        "checkpoint": str(ck.resolve()), "checkpoint_bytes": st.st_size,
        "checkpoint_mtime": int(st.st_mtime),
        "data_format": d.get("format", "pianovam"), "data_root": str(d.get("root")),
        "split": split, "max_frames": int(max_frames or 0), "fps": cfg["labels"]["fps"],
        "warp": [kb["warp_width"], kb["warp_height"]], "decode_height": kb.get("decode_height", 0),
    }


class RollsCache:
    """Per-recording probability rolls in one directory (``None`` = disabled)."""

    def __init__(self, directory: Optional[str], meta: Dict[str, Any]):
        self.dir = Path(directory) if directory else None
        if self.dir is None:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            meta_path = self.dir / "meta.json"
            old = None
            if meta_path.exists():
                try:
                    old = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    old = None
            if old != meta:
                stale = list(self.dir.glob("*.npz"))
                if stale:
                    print(f"rolls in {self.dir} were made from other settings; recomputing them")
                for f in stale:
                    f.unlink()
                meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        except OSError as e:
            print(f"cannot keep model outputs in {self.dir} ({e}); running without saving them")
            self.dir = None

    # The cache must never change results: a file that cannot be read is simply
    # recomputed, and a save that fails (e.g. a full disk) only switches saving
    # off -- it must not make the caller skip the recording.
    def load(self, record: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.dir is None:
            return None
        f = self.dir / f"{record}.npz"
        if not f.exists():
            return None
        try:
            with np.load(f) as z:
                return z["onset"].astype(np.float64), z["frame"].astype(np.float64)
        except Exception as e:  # corrupt or truncated file
            print(f"could not read {f} ({e}); running the model again")
            return None

    def save(self, record: str, onset_p: np.ndarray, frame_p: np.ndarray) -> None:
        if self.dir is None:
            return
        tmp = self.dir / f"{record}.partial.npz"
        try:
            np.savez_compressed(tmp, onset=onset_p.astype(np.float32),
                                frame=frame_p.astype(np.float32))
            tmp.replace(self.dir / f"{record}.npz")
        except OSError as e:
            print(f"could not save model outputs in {self.dir} ({e}); continuing without saving")
            try:
                tmp.unlink()
            except OSError:
                pass
            self.dir = None
