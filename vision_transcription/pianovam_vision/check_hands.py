"""Check that the PianoVAM hand data is complete and consistent; build the skeleton cache.

    python -m pianovam_vision.check_hands --config configs/vam_hands.yaml            # labels only (fast)
    python -m pianovam_vision.check_hands --config configs/vam_hands.yaml --skeleton  # + landmarks, cache

Per recording of the chosen splits:
  * Fingering/<rec>.tsv exists and its first five columns equal TSV/<rec>.tsv row
    for row (the hand/finger labels belong to exactly our notes); share of Noinfo.
  * Fingering_GT/<rec>.tsv, where present, lines up with the first rows.
  * Handskeleton/<rec>.json exists; with --skeleton it is read and projected
    into the keyboard strip (written to ``data.skeleton_cache``, which training
    and evaluation then read in a fraction of a second), and checked:
      - it covers the video (frame counts within 2 label frames of the strip
        cache, when that is built);
      - how often each hand is detected;
      - the hand tags: the nearest-fingertip hand must agree with the dataset's
        own labels for most notes with the tags swapped (``hands.swap_tags``);
        a recording where it does not is listed as a problem.
Ends with one verdict line; exit code 1 if anything blocks training.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import hands as H
from .config import load_config
from .labels import read_tsv
from .metadata import recordings_from_cfg


def video_frames(cfg: Dict[str, Any], rec):
    """Label frames of the video: from the strip cache, else the video header."""
    from .strip_cache import try_open_cached, video_path_for

    reader, _ = try_open_cached(cfg, rec, 0)
    if reader is not None:
        return len(reader)
    vp = video_path_for(cfg, rec)
    if not vp.exists():
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(vp))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not n or not fps:
        return None
    return int(np.floor(n / (fps / float(cfg["labels"]["fps"]))))


def check_one(cfg: Dict[str, Any], rec, skeleton: bool) -> Dict[str, Any]:
    r: Dict[str, Any] = {"record": rec.record_time, "split": rec.split, "problems": [], "notes": []}
    root = Path(cfg["data"]["root"])
    tsv = root / cfg["data"]["tsv_dir"] / f"{rec.record_time}.tsv"
    fp = H.fingering_path(cfg, rec.record_time)
    if not fp.exists():
        r["problems"].append("no Fingering/ file")
        return r
    lab = H.read_fingering(fp, "key_offset")
    r["n"] = len(lab)
    r["noinfo"] = sum(c < 0 for _, c in lab) / max(1, len(lab))
    if tsv.exists():
        ref = read_tsv(tsv, "key_offset")
        mine = sorted((round(n.onset, 5), n.pitch) for n, _ in lab)
        theirs = sorted((round(n.onset, 5), n.pitch) for n in ref)
        if mine != theirs:
            r["problems"].append(f"Fingering/ notes differ from TSV/ ({len(mine)} vs {len(theirs)} rows)")
    else:
        r["problems"].append("no TSV/ file")
    gp = H.fingering_path(cfg, rec.record_time, gt=True)
    if gp.exists():
        g = H.read_fingering(gp, "key_offset")
        ok = all(abs(a.onset - b.onset) < 1e-4 and a.pitch == b.pitch
                 for (a, _), (b, _) in zip(g, lab))
        r["gt"] = len(g)
        if not ok:
            r["problems"].append("Fingering_GT/ rows do not line up with Fingering/")
    cp = H.skeleton_cache_path(cfg, rec.record_time)
    if not H.skeleton_json_path(cfg, rec.record_time).exists() and not (cp and cp.exists()):
        r["problems"].append("no Handskeleton/ file")
        return r
    if not skeleton:
        return r
    try:
        track = H.load_skeleton(cfg, rec, apply_offset=False)
    except Exception as e:                                   # noqa: BLE001
        r["problems"].append(f"skeleton unreadable: {e}")
        return r
    r["skel_frames"] = len(track)
    r["det_left"], r["det_right"] = (float(v) for v in track.present.mean(0))
    n_video = video_frames(cfg, rec)
    if n_video is not None:
        r["video_frames"] = n_video
        if abs(n_video - len(track)) > 2:
            # Not fatal by itself (the alignment below decides), but it means the
            # landmarks were computed on another version of the video.
            r["notes"].append(f"skeleton {len(track)} frames vs video {n_video}")
    # Time alignment of the landmarks with the video, without labels.
    dist, hit = H.alignment_curve(track, [n for n, _ in lab], cfg["labels"]["fps"], H.ALIGN_SHIFTS)
    if not np.isnan(dist).all():
        r["dist"], r["hit"] = dist, hit
        r["shift"] = int(H.ALIGN_SHIFTS[int(np.nanargmin(dist))])
    # Tag check against the dataset's own labels (first 2,000 labelled notes).
    labelled = [(n, c) for n, c in lab if c >= 0][:2000]
    if labelled:
        notes = [n for n, _ in labelled]
        cls, _ = H.decide(notes, H.fingertip_scores(notes, track, cfg["labels"]["fps"], 1, roi=False))
        dec = cls >= 0
        agree = float(np.mean([H.class_hand(int(a)) == H.class_hand(b)
                               for a, (_, b), d in zip(cls, labelled, dec) if d])) if dec.any() else 0.0
        r["tag_agree"] = agree
        if agree < 0.8:
            r["problems"].append(f"nearest-fingertip hand agrees with the dataset labels for only "
                                 f"{100 * agree:.0f}% of notes (hand tags the other way round here?)")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="train,ext-train,valid,test")
    ap.add_argument("--skeleton", action="store_true",
                    help="also read the landmarks: checks + build data.skeleton_cache")
    ap.add_argument("--workers", type=int, default=1, help="recordings read in parallel")
    ap.add_argument("--write_offsets", default=None,
                    help="with --skeleton: write the landmark time offsets (JSON) here; "
                         "point data.skeleton_offsets at it")
    ap.add_argument("--min_offset", type=int, default=2,
                    help="offsets smaller than this many frames are written as 0 (noise)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    splits = [s.strip() for s in args.splits.split(",")]
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in recordings_from_cfg(cfg) if r.split in splits and r.record_time not in excl]
    print(f"checking hand data of {len(recs)} recordings ({','.join(splits)}) in {cfg['data']['root']}"
          + (f"; skeleton cache: {cfg['data'].get('skeleton_cache') or '(none: set data.skeleton_cache)'}"
             if args.skeleton else ""))

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(check_one, [cfg] * len(recs), recs, [args.skeleton] * len(recs)))
    else:
        results = []
        for rec in recs:
            results.append(check_one(cfg, rec, args.skeleton))
            r = results[-1]
            extra = ""
            if "skel_frames" in r:
                extra = (f" | skeleton {r['skel_frames']} fr, detected L {100 * r['det_left']:.0f}% "
                         f"R {100 * r['det_right']:.0f}%, tags {100 * r.get('tag_agree', 0):.0f}% agree"
                         + (f", fingertips fit best at {r['shift']:+d} fr" if "shift" in r else ""))
                if r["notes"]:
                    extra += " (" + "; ".join(r["notes"]) + ")"
            print(f"  {r['record']} [{r['split']}] {r.get('n', 0)} notes, Noinfo "
                  f"{100 * r.get('noinfo', 0):.1f}%" + (f", {r['gt']} manual" if r.get("gt") else "")
                  + extra + ("  PROBLEM: " + "; ".join(r["problems"]) if r["problems"] else ""),
                  flush=True)

    by_split = Counter()
    noinfo = Counter()
    for r in results:
        by_split[r["split"]] += r.get("n", 0)
        noinfo[r["split"]] += r.get("n", 0) * r.get("noinfo", 0)
    print("\nnotes with a hand/finger label per split:")
    for s in splits:
        if by_split[s]:
            print(f"  {s:10s} {by_split[s]:7d} notes, {100 * (1 - noinfo[s] / by_split[s]):.1f}% labelled")
    gt = [(r["record"], r["split"], r["gt"]) for r in results if r.get("gt")]
    print(f"manual labels (Fingering_GT): {sum(g for *_, g in gt)} notes in {len(gt)} recordings: "
          + ", ".join(f"{rt} [{sp}]" for rt, sp, _ in gt))
    unseen = [g for g in gt if g[1] in ("valid", "test")]
    print(f"  of which unseen by training (valid/test): {sum(g for *_, g in unseen)} notes")
    if args.skeleton:
        ag = [r["tag_agree"] for r in results if "tag_agree" in r]
        if ag:
            print(f"hand tags (swapped={bool((cfg.get('hands') or {}).get('swap_tags', True))}): nearest "
                  f"fingertip agrees with the dataset labels for {100 * np.mean(ag):.1f}% of notes "
                  f"(min {100 * min(ag):.1f}% over recordings)")
        al = [r for r in results if "shift" in r]
        if al:
            # A finger is on its key slightly before the onset, so in-sync recordings
            # peak at the same small shift; the offset is measured from their median.
            med = int(np.median([r["shift"] for r in al]))
            i_med = int(np.where(H.ALIGN_SHIFTS == med)[0][0])
            offs = {r["record"]: H.choose_offset(r["dist"], med, args.min_offset) for r in al}
            moved = {k: v for k, v in offs.items() if v}
            print(f"landmark timing: fingertips fit the pressed keys best at {med:+d} frames in most "
                  f"recordings; {len(moved)} recording(s) clearly elsewhere (>= {args.min_offset} "
                  f"frames)" + (":" if moved else ""))
            for r in al:
                o = offs[r["record"]]
                if o:
                    j = int(np.where(H.ALIGN_SHIFTS == med + o)[0][0])
                    print(f"  {r['record']} [{r['split']}] offset {o:+d} frames "
                          f"({1000 * o / cfg['labels']['fps']:+.0f} ms): fingertip-to-key distance "
                          f"{r['dist'][i_med]:.3f} -> {r['dist'][j]:.3f} keys, fingertip on the key "
                          f"{100 * r['hit'][i_med]:.0f}% -> {100 * r['hit'][j]:.0f}%"
                          + (f"  [{'; '.join(r['notes'])}]" if r["notes"] else ""))
            if args.write_offsets:
                import json

                Path(args.write_offsets).parent.mkdir(parents=True, exist_ok=True)
                with open(args.write_offsets, "w", encoding="utf-8") as f:
                    json.dump(dict(sorted(offs.items())), f, indent=1)
                print(f"  offsets -> {args.write_offsets} (set data.skeleton_offsets to use them)")
            elif moved and not cfg["data"].get("skeleton_offsets"):
                print("  -> run again with --write_offsets and set data.skeleton_offsets, or these "
                      "recordings train and score with landmarks out of sync")
    bad = [r for r in results if r["problems"]]
    if bad:
        print(f"\nPROBLEMS in {len(bad)} recording(s):")
        for r in bad:
            print(f"  {r['record']} [{r['split']}]: " + "; ".join(r["problems"]))
        print("VERDICT: fix the above (scripts/download_hand_data.py fetches missing files)")
        sys.exit(1)
    print("\nVERDICT: hand data complete and consistent"
          + ("" if args.skeleton else " (labels; add --skeleton to check the landmarks too)"))


if __name__ == "__main__":
    main()
