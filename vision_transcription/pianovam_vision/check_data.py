"""Verify dataset layout and report what's available per split.

    python -m pianovam_vision.check_data --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .config import load_config
from .metadata import load_recordings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["data"]["root"])
    meta = root / cfg["data"]["metadata"]
    if not meta.exists():
        print(f"[FATAL] metadata not found: {meta}")
        return

    recs = load_recordings(meta)
    print(f"metadata: {meta}  ({len(recs)} recordings)")
    print("split counts:", dict(Counter(r.split for r in recs)))

    missing = 0
    for r in recs:
        vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        tp = r.tsv_path(root, cfg["data"]["tsv_dir"])
        mp = r.midi_path(root, cfg["data"]["midi_dir"])
        miss = [str(p) for p in (vp, tp) if not p.exists()]
        if miss:
            missing += 1
            print(f"  [missing] {r.record_time} ({r.split}): {miss}")
    if missing == 0:
        print("OK: all video+tsv files for every recording are present.")
    else:
        print(f"{missing} recording(s) have missing files (see above).")


if __name__ == "__main__":
    main()
