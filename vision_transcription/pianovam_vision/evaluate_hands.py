"""Score which-hand / which-finger decisions on PianoVAM (see HANDS.md).

    python -m pianovam_vision.evaluate_hands --config configs/vam_hands.yaml --split test \
        [--checkpoint runs/vam_hands/best.pt] \
        [--pred KeyWise=out/E1s2_vam_full_test --pred V2N=/path/to/V2N/pianovam_test] \
        [--csv results/hands_test.csv]  [decode.onset_threshold=.. decode.frame_threshold=..]

Methods (``--methods``, default: all that the inputs allow):
  pitch  no vision: below C4 left hand, else right (hand only)
  tip    nearest fingertip to the key centre at the onset frame (the classic
         MediaPipe rule, tags swapped)
  roi    nearest fingertip to the key's real surface, averaged over the note's
         first ``hands.label_frames`` frames (the key-ROI rule)
  ayf    At Your Fingertips' rule (Moryossef et al.): Gaussian of each fingertip
         around the key centre, normalised over fingers, first frame weighted most
  auto   the dataset's own automatic labels (Fingering/), given notes only
  model  our hand branch (needs --checkpoint with model.hands)

Two settings:
  given notes   the reference notes are known; only "which hand / finger" is
                judged. Accuracy is over ALL notes: a note without a decision
                (no hand detected, Noinfo) counts as wrong; coverage is shown.
  end-to-end    transcribed notes (the checkpoint's own, and/or MIDI folders
                via --pred NAME=DIR) are matched to the reference within 50 and
                100 ms (mir_eval's onset matching); Note F1 counts a matched note,
                Note+Hand F1 also needs the right hand, +Finger the right finger.

Two label sets:
  manual     Fingering_GT/ (the first 150-300 notes of 11 recordings): the
             real ground truth. Only valid 2024-02-15_20-07-54 and test
             2024-02-17_21-44-37, 2024-09-02_14-10-41 are unseen by a model
             trained on train + ext-train; rule methods can use all 11
             (``--split train,ext-train,valid,test``).
  automatic  Fingering/ (every note of the split; Noinfo notes left out). Made
             from MediaPipe by the dataset authors (99.2 % hand / 95.7 % finger
             correct where labelled), so scoring a skeleton rule against them
             is partly circular: use them for the size, the manual ones for truth.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import hands as H
from .config import config_from_checkpoint, load_config
from .labels import Note
from .metadata import recordings_from_cfg
from .metrics import _pitchwise_matching

ALL_METHODS = ("pitch", "tip", "roi", "ayf", "auto", "model")
E2E_TOLERANCES = (0.05, 0.10)          # onset tolerances of the paper (50 / 100 ms)
METHOD_NAMES = {"pitch": "pitch split (C4)", "tip": "fingertip rule (centre)",
                "roi": "fingertip rule (key ROI)", "ayf": "At Your Fingertips rule",
                "auto": "dataset labels",
                "model": "KeyWise-hands"}


def mcnemar_p(a_ok: np.ndarray, b_ok: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired right/wrong outcomes."""
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_references(cfg, rec) -> Tuple[List[Note], np.ndarray, Optional[np.ndarray]]:
    """Reference notes in file order, their automatic classes, and the manual
    classes of the first rows (None when the recording has no Fingering_GT)."""
    fp = H.fingering_path(cfg, rec.record_time)
    if not fp.exists():
        raise FileNotFoundError(f"no {fp}: run scripts/download_hand_data.py")
    auto = H.read_fingering(fp, cfg["labels"]["offset_field"])
    notes = [n for n, _ in auto]
    gt = None
    gp = H.fingering_path(cfg, rec.record_time, gt=True)
    if gp.exists():
        g = H.read_fingering(gp, cfg["labels"]["offset_field"])
        ok = len(g) <= len(notes) and all(
            abs(a.onset - b.onset) < 1e-4 and a.pitch == b.pitch for (a, _), b in zip(g, notes))
        if ok:
            gt = np.array([c for _, c in g], dtype=np.int64)
        else:
            print(f"  {rec.record_time}: WARNING Fingering_GT rows do not line up with "
                  f"Fingering/ -- manual labels skipped")
    return notes, np.array([c for _, c in auto], dtype=np.int64), gt


def method_scores(method: str, notes: List[Note], cfg, ctx) -> Optional[np.ndarray]:
    hc = cfg.get("hands") or {}
    fps, lf = cfg["labels"]["fps"], int(hc.get("label_frames", 8))
    if method == "pitch":
        return H.pitch_split(notes, int(hc.get("split_pitch", 60)))
    if method in ("tip", "roi"):
        return H.fingertip_scores(notes, ctx["track"], fps, lf, roi=(method == "roi"))
    if method == "ayf":
        return H.ayf_scores(notes, ctx["track"], fps)
    if method == "model":
        return H.probs_scores(notes, ctx["probs"], fps, lf)
    raise ValueError(method)


def hands_of(cls: np.ndarray) -> np.ndarray:
    return np.where(cls < 0, -1, cls // 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", help="split(s), comma separated")
    ap.add_argument("--checkpoint", default=None, help="a model with a hand branch")
    ap.add_argument("--methods", default=None, help=f"subset of {','.join(ALL_METHODS)}")
    ap.add_argument("--pred", action="append", default=[],
                    help="NAME=DIR of <record_time>.mid transcriptions to score end-to-end")
    ap.add_argument("--chord_matching", action="store_true",
                    help="different fingers for the notes of one chord (Hungarian)")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="score only the first N frames of each video (0 = all)")
    ap.add_argument("--csv", default=None, help="per-recording rows")
    ap.add_argument("--save_dir", default=None,
                    help="write the end-to-end decisions (TSV per recording) here")
    ap.add_argument("--save_probs", default=None,
                    help="write every reference note's class probabilities (model) and rule "
                         "scores here, one <record_time>.npz each, for hand_report")
    ap.add_argument("--save_probs_manual_only", action="store_true",
                    help="--save_probs only for recordings with manual labels")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ckpt = model = None
    if args.checkpoint:
        import torch

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = config_from_checkpoint(ckpt, args.config, args.overrides)
        if not (cfg["model"].get("hands") or {}).get("enable", False):
            raise SystemExit(f"{args.checkpoint} has no hand branch (model.hands)")
    else:
        cfg = load_config(args.config, args.overrides)
    hc = cfg.get("hands") or {}
    fps, lf = cfg["labels"]["fps"], int(hc.get("label_frames", 8))
    methods = [m.strip() for m in (args.methods or ",".join(ALL_METHODS)).split(",") if m.strip()]
    if "model" in methods and ckpt is None:
        methods.remove("model")
        if args.methods:
            print("no --checkpoint: skipping method 'model'")
    preds = {}
    for p in args.pred:
        name, _, d = p.partition("=")
        if not d:
            raise SystemExit(f"--pred wants NAME=DIR, got {p!r}")
        preds[name] = Path(d)
    chord = args.chord_matching or bool(hc.get("chord_matching", False))

    splits = [s.strip() for s in args.split.split(",")]
    excl = set(cfg["data"].get("exclude_records", []) or [])
    recs = [r for r in recordings_from_cfg(cfg) if r.split in splits and r.record_time not in excl]
    print(f"hands evaluation on {','.join(splits)}: {len(recs)} recordings | methods "
          f"{','.join(methods)} | read-out window {lf} frames"
          + (" | chord matching" if chord else ""))

    # given[label set][method] -> list of per-note right/wrong (hand, finger, decided)
    given = defaultdict(lambda: defaultdict(lambda: {"hand": [], "finger": [], "dec": []}))
    # e2e[label set][(source, method)] -> counts
    e2e = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    rows = []
    device = None
    for rec in recs:
        try:
            notes, auto, gt = load_references(cfg, rec)
        except FileNotFoundError as e:
            print(f"  {rec.record_time}: [skipped] {e}")
            continue
        ctx = {}
        if any(m in methods for m in ("tip", "roi", "ayf")) or args.save_probs or (
                "model" in methods and (cfg["model"].get("hands") or {}).get("skeleton", True)):
            ctx["track"] = H.load_skeleton(cfg, rec)
        sources: Dict[str, List[Note]] = {}
        if "model" in methods:
            import torch

            from .decode import decode_with_cfg
            from .infer import predict_outputs
            from .model import build_model
            from .strip_cache import open_reader

            if model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_model(cfg).to(device)
                model.load_state_dict(ckpt["model"])
                model.eval()
            try:
                reader = open_reader(cfg, rec, args.max_frames)
                n = len(reader)
                skel = (ctx["track"].fit(n).model_input(np.arange(n))
                        if model.hand_branch.skel is not None else None)
                o = predict_outputs(model, reader, cfg, device, skel)
            except Exception as e:                          # noqa: BLE001
                # Like evaluate.py: an unreadable video is skipped, not fatal. The
                # recording is left out of EVERY method, so all stay paired.
                print(f"  {rec.record_time}: [skipped] cannot run the model ({e})")
                continue
            ctx["probs"] = o["hands"]
            sources["model"] = decode_with_cfg(o["onset"], o["frame"], o["velocity"], cfg)
        for name, d in preds.items():
            f = d / f"{rec.record_time}.mid"
            if not f.exists():
                print(f"  {rec.record_time}: no {f}, {name} skipped here")
                continue
            from .score_midi import load_midi_notes
            sources[name] = load_midi_notes(f)

        # Restrict everything to the frames that were looked at (--max_frames).
        t_max = (args.max_frames / fps) if args.max_frames else float("inf")
        keep = np.array([n.onset < t_max for n in notes], dtype=bool)
        if not keep.all():
            notes = [n for n, k in zip(notes, keep) if k]
            auto = auto[keep]
            if gt is not None:
                gt = gt[:min(len(gt), int(keep.sum()))]
        for k in list(sources):
            sources[k] = [n for n in sources[k] if n.onset < t_max]

        # ---- given notes
        label_sets = [("automatic", auto, np.ones(len(notes), bool) & (auto >= 0))]
        if gt is not None and len(gt):
            mask = np.zeros(len(notes), bool)
            mask[:len(gt)] = True
            full_gt = np.full(len(notes), -1)
            full_gt[:len(gt)] = gt
            label_sets.insert(0, ("manual", full_gt, mask))
        decided = {}
        for m in methods:
            if m == "auto":
                cls = auto.copy()
            else:
                cls, _ = H.decide(notes, method_scores(m, notes, cfg, ctx), chord, joint=(m == "ayf"))
            decided[m] = cls
            for lname, lab, mask in label_sets:
                if m == "auto" and lname == "automatic":
                    continue
                c, y = cls[mask], lab[mask]
                g = given[lname][m]
                g["hand"] += list((c >= 0) & (hands_of(c) == hands_of(y)))
                g["finger"] += list((c >= 0) & (c == y))
                g["dec"] += list(c >= 0)
                rows.append({"record": rec.record_time, "split": rec.split, "setting": "given",
                             "labels": lname, "notes": "reference", "method": m, "n": int(mask.sum()),
                             "coverage": round(float((c >= 0).mean()), 4) if len(c) else 0.0,
                             "hand_acc": round(float(((c >= 0) & (hands_of(c) == hands_of(y))).mean()), 4)
                             if len(c) else 0.0,
                             "finger_acc": round(float(((c >= 0) & (c == y)).mean()), 4) if len(c) else 0.0})

        if args.save_probs and (gt is not None or not args.save_probs_manual_only):
            from .hand_notes import NO_GT, make_noteset, rule_scores, save_note_npz

            full_gt = np.full(len(notes), NO_GT)
            if gt is not None:
                full_gt[:len(gt)] = gt
            ns = make_noteset(notes, cfg, None, rec.corners, auto, full_gt,
                              {"record_time": rec.record_time, "split": rec.split,
                               "performer": rec.performer}, ctx.get("track"))
            from .hand_notes import session_of
            ns.meta["session"] = session_of(cfg, rec.record_time)
            probs = {}
            if "probs" in ctx:
                probs["p_frame"] = H.softmax_rows(H.probs_scores(ns.notes, ctx["probs"], fps, lf))
            save_note_npz(Path(args.save_probs) / f"{rec.record_time}.npz", ns, probs,
                          rule_scores(ns, cfg) if ns.track is not None else {}, {"seed": 1})

        # ---- end-to-end
        for src, est in sources.items():
            for m in methods:
                if m == "auto":
                    continue
                cls, conf = H.decide(est, method_scores(m, est, cfg, ctx), chord, joint=(m == "ayf"))
                if args.save_dir and m in ("model", "roi"):
                    H.write_hand_tsv(Path(args.save_dir) / f"{src}_{m}" / f"{rec.record_time}.tsv",
                                     est, cls, conf)
                for lname, lab, mask in label_sets:
                    if lname == "manual":
                        n_gt = int(mask.sum())
                        t_end = ((notes[n_gt - 1].onset + notes[n_gt].onset) / 2
                                 if n_gt < len(notes) else float("inf"))
                        ref_idx = np.arange(n_gt)
                        est_idx = np.array([i for i, n in enumerate(est) if n.onset < t_end], dtype=int)
                    else:
                        ref_idx = np.arange(len(notes))
                        est_idx = np.arange(len(est))
                    ref_sub = [notes[i] for i in ref_idx]
                    est_sub = [est[i] for i in est_idx]
                    unl_ref = int((lab[ref_idx] < 0).sum())
                    for tol in E2E_TOLERANCES:
                        pairs = _pitchwise_matching(ref_sub, est_sub, tol, None, 0.05)
                        c = e2e[lname][(src, m, int(round(tol * 1000)))]
                        c["ref"] += len(ref_sub) - unl_ref
                        c["est"] += len(est_sub)
                        for ri, ei in pairs:
                            y = lab[ref_idx[ri]]
                            if y < 0:
                                c["est"] -= 1      # matched an unlabelled note: not scored
                                continue
                            p = cls[est_idx[ei]]
                            c["match"] += 1
                            c["hand"] += int(p >= 0 and p // 5 == y // 5)
                            c["finger"] += int(p >= 0 and p == y)
        print(f"  {rec.record_time} [{rec.split}]: {len(notes)} notes"
              + (f", {len(gt)} manual" if gt is not None else "")
              + (f" | end-to-end: {', '.join(f'{k} {len(v)}' for k, v in sources.items())}"
                 if sources else ""), flush=True)

    # ---- report
    def f1(tp, n_est, n_ref):
        p, r = tp / max(1, n_est), tp / max(1, n_ref)
        return 100 * (2 * p * r / (p + r) if p + r > 0 else 0.0)

    for lname in ("manual", "automatic"):
        if lname not in given:
            continue
        res = given[lname]
        n = len(next(iter(res.values()))["hand"])
        print(f"\n=== given notes: {lname} labels ({n} notes) -- accuracy over all notes, % ===")
        print(f"  {'method':26s} {'coverage':>9} {'hand':>7} {'hand+finger':>12}")
        for m in methods:
            if m not in res:
                continue
            r = {k: np.array(v, bool) for k, v in res[m].items()}
            fing = "" if m == "pitch" else f"{100 * r['finger'].mean():12.1f}"
            print(f"  {METHOD_NAMES[m]:26s} {100 * r['dec'].mean():9.1f} {100 * r['hand'].mean():7.1f} {fing:>12}")
        present = [m for m in methods if m in res]
        pairs = [(a, b) for a, b in (("roi", "tip"), ("roi", "ayf"), ("roi", "auto"), ("model", "roi"),
                                     ("model", "ayf"), ("model", "auto"), ("roi", "pitch"))
                 if a in present and b in present]
        if pairs:
            print("  exact McNemar p (paired, same notes):")
            for a, b in pairs:
                ra = {k: np.array(v, bool) for k, v in res[a].items()}
                rb = {k: np.array(v, bool) for k, v in res[b].items()}
                ph = mcnemar_p(ra["hand"], rb["hand"])
                pf = mcnemar_p(ra["finger"], rb["finger"]) if "pitch" not in (a, b) else float("nan")
                print(f"    {a} vs {b}: hand p={ph:.4f}" + ("" if math.isnan(pf) else f", hand+finger p={pf:.4f}"))

    for lname in ("manual", "automatic"):
        if lname not in e2e:
            continue
        ms = [int(round(t * 1000)) for t in E2E_TOLERANCES]
        print(f"\n=== end-to-end: {lname} labels, F1 % (onset within {' / '.join(map(str, ms))} ms) ===")
        print(f"  {'notes from':14s} {'hands by':26s} {'Note':>11} {'+Hand':>11} {'+Hand+Finger':>13}")
        keys = sorted({(s, m) for s, m, _ in e2e[lname]})
        for src, m in keys:
            cs = [e2e[lname][(src, m, t)] for t in ms]
            note = " / ".join(f"{f1(c['match'], c['est'], c['ref']):4.1f}" for c in cs)
            hand = " / ".join(f"{f1(c['hand'], c['est'], c['ref']):4.1f}" for c in cs)
            fing = "" if m == "pitch" else " / ".join(
                f"{f1(c['finger'], c['est'], c['ref']):4.1f}" for c in cs)
            print(f"  {src:14s} {METHOD_NAMES[m]:26s} {note:>11} {hand:>11} {fing:>13}")
            for t, c in zip(ms, cs):
                rows.append({"record": "ALL", "split": ",".join(splits), "setting": f"end-to-end@{t}",
                             "labels": lname, "notes": src, "method": m, "n": c["ref"],
                             "note_f1": round(f1(c["match"], c["est"], c["ref"]), 2),
                             "hand_f1": round(f1(c["hand"], c["est"], c["ref"]), 2),
                             "finger_f1": round(f1(c["finger"], c["est"], c["ref"]), 2)})

    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))   # first-seen order
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nrows -> {args.csv}")


if __name__ == "__main__":
    main()
