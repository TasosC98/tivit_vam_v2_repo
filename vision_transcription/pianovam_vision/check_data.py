"""Verify dataset layout and report what's available per split.

    python -m pianovam_vision.check_data --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .config import load_config
from .labels import reference_path
from .metadata import recordings_from_cfg


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

    recs = recordings_from_cfg(cfg)
    print(f"metadata: {meta}  ({len(recs)} recordings, "
          f"format={cfg['data'].get('format', 'pianovam')})")
    print("split counts:", dict(Counter(r.split for r in recs)))

    missing = missing_video = missing_label = 0
    for r in recs:
        vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        lp = reference_path(r, cfg)
        miss = [str(p) for p in (vp, lp) if not p.exists()]
        if miss:
            missing += 1
            missing_video += not vp.exists()
            missing_label += not lp.exists()
            print(f"  [missing] {r.record_time} ({r.split}): {miss}")
    if missing == 0:
        print("OK: all video+label files for every recording are present.")
    else:
        print(f"{missing} recording(s) have missing files "
              f"({missing_video} video, {missing_label} label). "
              f"These are skipped automatically at train/eval time.")


if __name__ == "__main__":
    main()
