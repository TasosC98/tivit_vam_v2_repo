"""Verify dataset layout and report what's available per split.

    python -m pianovam_vision.check_data --config configs/default.yaml

``--probe`` also opens every video (needs decord) and reports resolution, fps
and duration, flags keyboard boxes that fall outside the frame (PianoYT boxes
are in the authors' pixel coordinates, so a download at another resolution
breaks them) and labels that run past the end of the video (sync/trim
problems), and totals the hours of usable video per split -- the numbers for
the paper's dataset table.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config
from .labels import read_reference, reference_path
from .metadata import recordings_from_cfg


def probe(recs, cfg, root: Path) -> None:
    import decord

    hours = defaultdict(float)
    notes = defaultdict(int)
    count = Counter()
    res, rates = Counter(), Counter()
    issues = []
    excl = set(cfg["data"].get("exclude_records", []) or [])
    print(f"\n{'record':>20} {'split':>6} {'WxH':>10} {'fps':>6} {'dur_s':>7} "
          f"{'label_end':>9}  notes")
    for r in recs:
        if r.record_time in excl:
            continue
        vp = r.video_path(root, cfg["data"]["video_dir"], cfg["data"]["video_ext"])
        lp = reference_path(r, cfg)
        if not (vp.exists() and lp.exists()):
            continue
        try:
            vr = decord.VideoReader(str(vp), num_threads=1)
            h, w = vr[0].shape[:2]
            fps = float(vr.get_avg_fps())
            dur = len(vr) / fps if fps else 0.0
        except Exception as e:
            issues.append(f"{r.record_time}: video does not decode ({e})")
            continue
        ref = read_reference(r, cfg)
        end = max((n.offset for n in ref), default=0.0)
        print(f"{r.record_time:>20} {r.split:>6} {f'{w}x{h}':>10} {fps:6.2f} "
              f"{dur:7.1f} {end:9.1f}  {len(ref)}")
        hours[r.split] += dur / 3600
        notes[r.split] += len(ref)
        count[r.split] += 1
        res[f"{w}x{h}"] += 1
        rates[f"{fps:.2f}"] += 1
        x, y = r.corners[:, 0], r.corners[:, 1]
        if x.min() < -2 or y.min() < -2 or x.max() > w + 2 or y.max() > h + 2:
            issues.append(f"{r.record_time}: keyboard box x[{x.min():.0f},{x.max():.0f}] "
                          f"y[{y.min():.0f},{y.max():.0f}] is outside the {w}x{h} frame "
                          f"-> wrong download resolution? (exclude or fix)")
        if ref and ref[-1].onset > dur + 1.0:
            issues.append(f"{r.record_time}: labels continue to {ref[-1].onset:.1f}s but "
                          f"the video is {dur:.1f}s long -> check sync/trim")

    print("\n=== usable data per split ===")
    for s in sorted(count):
        print(f"  {s:>8}: {count[s]:4d} videos, {hours[s]:6.2f} h, {notes[s]:8d} notes")
    print(f"  resolutions: {dict(res.most_common())}")
    print(f"  frame rates: {dict(rates.most_common())}")
    print(f"\n{len(issues)} issue(s):" if issues else "\nno issues found")
    for s in issues:
        print(f"  - {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", action="store_true",
                    help="open every video: resolution/fps/duration/box checks + hours")
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

    if args.probe:
        probe(recs, cfg, root)


if __name__ == "__main__":
    main()
