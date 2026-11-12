"""Download PianoVAM's hand data from Hugging Face into the dataset folder.

    python scripts/download_hand_data.py --root /raid_storage/data_achatzigiannis/PianoVAM_v1.0
    python scripts/download_hand_data.py --root ... --skeleton      # + Handskeleton/ (12 GB)

Fetches Fingering/ (automatic hand + finger per note, 106 files, 23 MB) and
Fingering_GT/ (manual labels, 11 files) of PianoVAM v1.2, and with --skeleton
the MediaPipe landmarks Handskeleton/ (108 files, 12.1 GB; most copies of the
dataset already have it). A file that already exists with the right size is
skipped, so the command can be re-run after an interruption. No login needed.
(The repository was renamed PianoVAM/PianoVAM_v1.0 -> PianoVAM/PianoVAM_v1.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "PianoVAM/PianoVAM_v1"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
FILES = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def listing(folder: str):
    with urllib.request.urlopen(API + folder, timeout=60) as r:
        return [e for e in json.load(r) if e.get("type") == "file"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder (holds TSV/, Video/, ...)")
    ap.add_argument("--skeleton", action="store_true", help="also Handskeleton/ (12 GB)")
    args = ap.parse_args()
    root = Path(args.root)
    if not (root / "TSV").is_dir():
        sys.exit(f"{root} has no TSV/ folder: is this the PianoVAM dataset folder?")
    folders = ["Fingering_GT", "Fingering"] + (["Handskeleton"] if args.skeleton else [])
    for folder in folders:
        entries = listing(folder)
        need = [e for e in entries if not ((root / e["path"]).exists()
                                           and (root / e["path"]).stat().st_size == e["size"])]
        gb = sum(e["size"] for e in need) / 1e9
        print(f"{folder}: {len(entries)} files, {len(need)} to download ({gb:.2f} GB)", flush=True)
        if need and shutil.disk_usage(root).free < sum(e["size"] for e in need) * 1.05:
            sys.exit(f"not enough free disk space in {root} for {folder}")
        for i, e in enumerate(need, 1):
            dst = root / e["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            urllib.request.urlretrieve(FILES + e["path"], tmp)
            if tmp.stat().st_size != e["size"]:
                tmp.unlink()
                sys.exit(f"size mismatch for {e['path']}: try again")
            os.replace(tmp, dst)
            if i % 20 == 0 or i == len(need):
                print(f"  {i}/{len(need)}", flush=True)
    print("done. Next: python -m pianovam_vision.check_hands --config configs/vam_hands.yaml --skeleton")


if __name__ == "__main__":
    main()
