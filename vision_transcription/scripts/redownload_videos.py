"""Find PianoYT videos that don't decode, and optionally re-download them.

Some PianoYT downloads are broken: the .mp4 exists but has no video stream
(decord fails with "cannot find video stream"). This script probes every video,
cross-references pianoyt.csv for the YouTube URL, reports the broken/missing
ones, and can re-fetch them with yt-dlp.

Run from vision_transcription/ with the venv active:

  # 1. just report which videos are missing or won't decode (+ their URLs):
  python scripts/redownload_videos.py --config configs/pianoyt.yaml

  # 2. actually re-download the broken/missing ones (needs yt-dlp on PATH):
  python scripts/redownload_videos.py --config configs/pianoyt.yaml --download

  # limit to one split, e.g. only the training videos:
  python scripts/redownload_videos.py --config configs/pianoyt.yaml --split train

For --download you need yt-dlp:  pip install -U yt-dlp

IMPORTANT: after re-downloading, verify the new files both DECODE and still line
up with the crop box (the box is in native-resolution pixels, so a different
download resolution can shift it):
  python -m pianovam_vision.check_data --config configs/pianoyt.yaml
  python -m pianovam_vision.preview     --config configs/pianoyt.yaml --record_time <id> --snap_to_onset --out_dir preview/
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pianovam_vision.config import load_config
from pianovam_vision.metadata import recordings_from_cfg


def decodes(path: Path) -> bool:
    """True if decord can open the file and it has at least one frame."""
    try:
        import decord
        vr = decord.VideoReader(str(path), num_threads=1)
        return len(vr) > 0
    except Exception:
        return False


def load_urls(csv_path: Path) -> dict[str, str]:
    """Map videoID -> YouTube URL from pianoyt.csv (id, url, split, ...)."""
    urls: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip("'").strip('"') for p in line.split(",")]
            if len(parts) >= 2:
                urls[parts[0]] = parts[1]
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--download", action="store_true",
                    help="re-download broken/missing videos with yt-dlp")
    ap.add_argument("--split", default=None,
                    help="only check this split (train/valid/test); default all")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    root = Path(d["root"])
    vdir = root / d["video_dir"]
    urls = load_urls(root / d["metadata"])

    recs = recordings_from_cfg(cfg)
    if args.split:
        recs = [r for r in recs if r.split == args.split]

    bad: list[tuple[str, Path, str]] = []
    for r in recs:
        vp = r.video_path(root, d["video_dir"], d["video_ext"])
        if not vp.exists():
            bad.append((r.record_time, vp, "missing"))
        elif not decodes(vp):
            bad.append((r.record_time, vp, "no video stream"))

    print(f"{len(recs)} recording(s) checked -> {len(bad)} broken/missing\n")
    for vid, _vp, why in bad:
        print(f"  {vid:>4}  {why:<16} {urls.get(vid, '(no URL in csv)')}")
    if not bad:
        print("\nAll videos decode. Nothing to do.")
        return
    if not args.download:
        print(f"\nRe-run with --download to fetch these {len(bad)} via yt-dlp.")
        return

    vdir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for vid, vp, _why in bad:
        url = urls.get(vid)
        if not url:
            print(f"[skip] {vid}: no URL in csv")
            fail += 1
            continue
        # best video + best audio, muxed to mp4, written to the exact expected name.
        cmd = ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
               "--no-playlist", "-o", str(vp), url]
        print(f"\n[dl] {vid} <- {url}")
        rc = subprocess.run(cmd).returncode
        if rc == 0 and decodes(vp):
            ok += 1
            print(f"     recovered: {vp.name}")
        else:
            fail += 1
            print(f"     FAILED for {vid} (video may be private/removed)")

    print(f"\ndone: {ok} recovered, {fail} still broken.")
    print("Verify decodability + crop before retraining:")
    print("  python -m pianovam_vision.check_data --config", args.config)


if __name__ == "__main__":
    main()
