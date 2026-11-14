"""Tables from the per-note outputs of ``hand_notes`` (and ``evaluate_hands --save_probs``).

    python -m pianovam_vision.hand_report --run keyrel=runs/hn/loro_keyrel \
        --run image=runs/hn/loro_image --run midi=runs/hn/loro_none \
        [--labels manual] [--main keyrel:seq] [--out results/hn_loro.txt] \
        [--csv results/hn_loro_notes.csv] [--curves results/hn_loro_curves]

A run directory holds fold directories ``<fold>_s<seed>/pred/<record>.npz``
(any depth); every seed of a run is one replicate. All runs must have scored
the same notes (the same scheme): rows are aligned by (recording, row).

Methods: for every run its decoders -- ``frame`` (frame read-out), ``seq``
(note-sequence model), ``frame+hmm`` / ``seq+hmm`` (the run's own
FingerPrior, Viterbi per hand) -- and the baselines stored with the notes:
``pitch`` (C4 split, hand only), ``tip`` and ``roi`` (fingertip rules), ``ayf``
(At Your Fingertips' rule) and ``auto`` (the dataset's labels).

Statistics (HANDS.md, part 5): accuracy over ALL evaluated notes (no decision =
wrong); mean +/- s.d. over seeds; 95 % intervals from a block bootstrap
(blocks of 10 consecutive notes resampled within each recording) and, for the
differences, also with the recordings resampled; per-recording comparisons with
an exact sign-flip test over recordings; McNemar only as a note-level aside.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import hands as H
from .labels import Note

BASELINES = ("pitch", "tip", "roi", "ayf", "auto")
DECODERS = ("seq", "frame", "seq+hmm", "frame+hmm")
COVERAGES = (100, 95, 90, 85, 80, 70, 60, 50)
BLOCK = 10


# ------------------------------------------------------------------ loading
def load_run(path: Path, sub: str = "pred") -> Dict[int, Dict[str, Dict]]:
    """{seed: {record: {arrays..., "prior": FingerPrior or None, "meta": dict}}}
    from <path>/**/<sub>/*.npz, or <path>/*.npz (evaluate_hands --save_probs)."""
    out: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    files = sorted(path.rglob(f"{sub}/*.npz")) or sorted(path.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {path}")
    for f in files:
        if f.name.endswith(".partial.npz"):
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(str(d.pop("meta")))
        seed = int(meta.get("seed", 1))
        rt = meta["record_time"]
        pj = f.parent.parent / "prior.json"
        d["prior"] = H.FingerPrior.from_dict(json.loads(pj.read_text())) if pj.exists() else None
        d["meta"] = meta
        if rt in out[seed]:
            raise SystemExit(f"{path}: {rt} evaluated twice for seed {seed} ({f})")
        out[seed][rt] = d
    return dict(out)


def notes_of(d: Dict) -> List[Note]:
    return [Note(float(a), float(b), int(p), 64) for a, b, p in zip(d["onset"], d["offset"], d["pitch"])]


def probs_from_scores(s: np.ndarray) -> np.ndarray:
    return H.softmax_rows(s)


def decode(d: Dict, decoder: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(classes, hand confidence, class confidence) of one decoder, or None."""
    notes = None
    if decoder in BASELINES:
        if decoder == "auto":
            c = d["auto"].astype(np.int64)
            return c, (c >= 0).astype(float), (c >= 0).astype(float)
        key = f"s_{decoder}"
        if key not in d:
            return None
        p = probs_from_scores(d[key])
        cls, ch = H.decide(notes_of(d), np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf),
                           joint=(decoder == "ayf"))
    else:
        base, _, hmm = decoder.partition("+")
        key = {"seq": "p_seq", "frame": "p_frame"}[base]
        if key not in d or not np.asarray(d[key]).any():
            return None
        p = np.asarray(d[key], np.float64)
        notes = notes_of(d)
        if hmm:
            if d.get("prior") is None:
                return None
            cls = H.viterbi_fingers(notes, p, d["prior"])
        else:
            cls, _ = H.decide(notes, np.log(np.clip(p, 1e-12, 1.0)))
        ch = np.where(cls >= 0, np.where(cls >= 5, p[:, 5:].sum(1), p[:, :5].sum(1)), 0.0)
    cc = np.where(cls >= 0, p[np.arange(len(p)), np.maximum(cls, 0)], 0.0)
    return cls, ch, cc


# ------------------------------------------------------------------ statistics
def block_ids(rec: np.ndarray) -> np.ndarray:
    """Block of 10 consecutive evaluated notes within each recording."""
    out = np.zeros(len(rec), np.int64)
    nxt = 0
    for r in dict.fromkeys(rec):
        idx = np.where(rec == r)[0]
        out[idx] = nxt + np.arange(len(idx)) // BLOCK
        nxt = out[idx].max() + 1
    return out


def bootstrap(vals: np.ndarray, rec: np.ndarray, n_boot: int = 2000, seed: int = 0,
              resample_recordings: bool = False) -> np.ndarray:
    """(n_boot, M) bootstrap means of the columns of ``vals`` (n, M): blocks of
    10 notes resampled within each recording (and the recordings themselves
    when ``resample_recordings``)."""
    vals = np.asarray(vals, np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    blk = block_ids(rec)
    nb = blk.max() + 1
    sums = np.zeros((nb, vals.shape[1]))
    np.add.at(sums, blk, vals)
    cnt = np.bincount(blk, minlength=nb).astype(np.float64)
    recs = list(dict.fromkeys(rec))
    rb = [np.unique(blk[rec == r]) for r in recs]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, vals.shape[1]))
    for b in range(n_boot):
        chosen = rng.integers(0, len(recs), len(recs)) if resample_recordings else range(len(recs))
        idx = np.concatenate([rb[i][rng.integers(0, len(rb[i]), len(rb[i]))] for i in chosen])
        out[b] = sums[idx].sum(0) / cnt[idx].sum()
    return out


def ci(boot: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_flip_p(diffs: Sequence[float]) -> float:
    """Exact two-sided sign-flip (permutation) test of mean difference over units."""
    d = np.asarray([x for x in diffs if abs(x) > 1e-12], np.float64)
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    else:
        signs = np.random.default_rng(0).choice((-1, 1), size=(20000, len(d)))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    b_, c_ = int((a & ~b).sum()), int((~a & b).sum())
    n = b_ + c_
    if n == 0:
        return 1.0
    k = min(b_, c_)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR (repeatable)")
    ap.add_argument("--labels", default="manual", choices=["manual", "automatic"])
    ap.add_argument("--main", default=None, help="RUN:DECODER compared with everything (default: first run, seq)")
    ap.add_argument("--decoders", default=",".join(DECODERS))
    ap.add_argument("--out", default=None, help="also write the report here")
    ap.add_argument("--csv", default=None, help="per-note decisions of every method")
    ap.add_argument("--curves", default=None, help="directory for precision-coverage CSV (+ PNG)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--pred_dir", default="pred",
                    help="prediction folder inside each fold (pred_<test> for hand_notes robust)")
    args = ap.parse_args(argv)

    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    runs = {}
    for r in args.run:
        name, _, d = r.partition("=")
        if not Path(d).exists():
            print(f"  (run {name}: no {d}; left out)")
            continue
        try:
            runs[name] = load_run(Path(d), args.pred_dir)
        except SystemExit as e:
            print(f"  (run {name}: {e}; left out)")
    if not runs:
        raise SystemExit("no run with results")
    # Runs still in progress: compare on the recordings that EVERY kept seed has
    # scored; a seed missing most of them is left out (reported).
    have = {(n, s): set(rr) for n, rs in runs.items() for s, rr in rs.items()}
    full = set.union(*have.values())
    for (n, s), got in sorted(have.items()):
        if got != full:
            if len(got) * 2 < len(full):
                print(f"  (run {n} seed {s}: only {len(got)}/{len(full)} recordings done; left out)")
                del runs[n][s]
            else:
                full &= got
    runs = {n: rs for n, rs in runs.items() if rs}
    recs = sorted(full)
    if not recs:
        raise SystemExit("no recording scored by every run")
    partial = any(set(rr) != set(recs) for rs in runs.values() for rr in rs.values())
    first = next(iter(runs.values()))
    seed0 = sorted(first)[0]
    ref = first[seed0]

    # evaluated notes: (recording, index into the npz arrays)
    sel = {}
    for rt in recs:
        d = ref[rt]
        sel[rt] = np.where(d["gt"] >= 0)[0] if args.labels == "manual" else np.where(d["auto"] >= 0)[0]
    recs = [rt for rt in recs if len(sel[rt])]          # recordings with evaluated notes
    if not recs:
        raise SystemExit(f"no notes with {args.labels} labels")
    rec_of = np.concatenate([[rt] * len(sel[rt]) for rt in recs])
    y = np.concatenate([(ref[rt]["gt"] if args.labels == "manual" else ref[rt]["auto"])[sel[rt]] for rt in recs])
    meta = {rt: ref[rt]["meta"] for rt in recs}
    n = len(y)
    if n == 0:
        raise SystemExit("no evaluated notes")

    # ---- decisions of every method: {method: [(cls, hand conf, class conf) per seed]}
    res: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    wanted = [x.strip() for x in args.decoders.split(",") if x.strip()]
    for name, rs in runs.items():
        for dec in wanted:
            per_seed = []
            for s in sorted(rs):
                parts = [decode(rs[s][rt], dec) for rt in recs]
                if any(p is None for p in parts):
                    per_seed = []
                    break
                per_seed.append(tuple(np.concatenate([p[i][sel[rt]] for p, rt in zip(parts, recs)])
                                      for i in range(3)))
            if per_seed:
                res[f"{name}:{dec}"] = per_seed
    for b in BASELINES:
        if b == "auto" and args.labels == "automatic":
            continue
        parts = [decode(ref[rt], b) for rt in recs]
        if all(p is not None for p in parts):
            res[b] = [tuple(np.concatenate([p[i][sel[rt]] for p, rt in zip(parts, recs)]) for i in range(3))]
    main_m = args.main or next((m for m in res if m.endswith(":seq")), next(iter(res)))
    if main_m not in res:
        raise SystemExit(f"--main {main_m} not available; have {sorted(res)}")

    def ok(m: str, finger: bool = True) -> np.ndarray:
        """(n,) correctness averaged over seeds."""
        acc = []
        for cls, _, _ in res[m]:
            acc.append((cls >= 0) & ((cls == y) if finger else (cls // 5 == y // 5)))
        return np.mean(acc, 0)

    def per_seed_acc(m: str, finger: bool = True) -> np.ndarray:
        return np.array([100 * (((c >= 0) & ((c == y) if finger else (c // 5 == y // 5))).mean())
                         for c, _, _ in res[m]])

    pianists = sorted({meta[r].get("performer", "?") for r in recs})
    if partial:
        say("  NOTE: some runs are not finished; everything below is on the recordings all runs have done")
    say(f"=== {args.labels} labels: {n} notes in {len(recs)} recordings, {len(pianists)} pianists "
        f"({', '.join(pianists)}) | runs: {', '.join(f'{k} ({len(v)} seeds)' for k, v in runs.items())}")
    say(f"    main method: {main_m}; accuracy over ALL notes (no decision = wrong), %; "
        f"95 % CI: block bootstrap (blocks of {BLOCK} notes within recordings)")

    # ---- 1. main table
    methods = list(res)
    finger_ok = {m: ok(m) for m in methods}
    hand_ok = {m: ok(m, False) for m in methods}
    boot_f = bootstrap(np.stack([finger_ok[m] for m in methods], 1), rec_of, args.n_boot)
    boot_h = bootstrap(np.stack([hand_ok[m] for m in methods], 1), rec_of, args.n_boot)
    say(f"\n  {'method':24s} {'seeds':>5s} {'answered':>8s}  {'hand':>22s}  {'hand + finger':>22s}")
    for j, m in enumerate(methods):
        cov = 100 * np.mean([(c >= 0).mean() for c, _, _ in res[m]])
        hs, fs = per_seed_acc(m, False), per_seed_acc(m)
        sd = lambda a: f" +/-{a.std(ddof=1):.1f}" if len(a) > 1 else "      "  # noqa: E731
        hc = ci(100 * boot_h[:, j]); fc = ci(100 * boot_f[:, j])
        fcol = "" if m == "pitch" else f"{fs.mean():5.1f}{sd(fs)} [{fc[0]:4.1f},{fc[1]:5.1f}]"
        say(f"  {m:24s} {len(res[m]):5d} {cov:8.1f}  {hs.mean():5.1f}{sd(hs)} [{hc[0]:4.1f},{hc[1]:5.1f}]  {fcol:>22s}")

    # ---- 2. paired differences with the main method
    say(f"\n  differences: {main_m} minus each method (hand + finger; hand), points; 95 % CI with blocks "
        f"(blocks + recordings); recordings better / worse; exact sign-flip p over recordings; McNemar p (notes, aside)")
    j0 = methods.index(main_m)
    boot_fr = bootstrap(np.stack([finger_ok[m] for m in methods], 1), rec_of, args.n_boot, 1, True)
    per_rec = {m: {rt: 100 * finger_ok[m][rec_of == rt].mean() for rt in recs} for m in methods}
    for j, m in enumerate(methods):
        if m == main_m or m == "pitch":
            continue
        dlt = 100 * (finger_ok[main_m] - finger_ok[m]).mean()
        c1 = ci(100 * (boot_f[:, j0] - boot_f[:, j]))
        c2 = ci(100 * (boot_fr[:, j0] - boot_fr[:, j]))
        dh = 100 * (hand_ok[main_m] - hand_ok[m]).mean()
        dr = [per_rec[main_m][rt] - per_rec[m][rt] for rt in recs]
        better, worse = sum(x > 1e-9 for x in dr), sum(x < -1e-9 for x in dr)
        pm = mcnemar_p(finger_ok[main_m] > 0.5, finger_ok[m] > 0.5)
        say(f"    vs {m:22s} {dlt:+5.1f} [{c1[0]:+5.1f},{c1[1]:+5.1f}] ([{c2[0]:+5.1f},{c2[1]:+5.1f}]); hand {dh:+4.1f}"
            f" | recordings {better}/{worse}, p={sign_flip_p(dr):.3f} | McNemar p={pm:.1e}")

    # ---- 3. per recording / pianist / session
    show = list(dict.fromkeys(
        [main_m] + [m for m in methods if m != main_m and (m in ("roi", "ayf", "auto") or ":" in m)][:7]))
    say(f"\n  per recording, hand + finger % ({' | '.join(show)})")
    for rt in recs:
        mk = rec_of == rt
        mt = meta[rt]
        say(f"    {rt} {mt.get('split', ''):9s} {mt.get('performer', ''):9s} {mt.get('session', ''):8s} n={mk.sum():3d}  "
            + "  ".join(f"{100 * finger_ok[m][mk].mean():5.1f}" for m in show))
    for key in ("performer", "session"):
        groups = sorted({meta[rt].get(key, "?") for rt in recs})
        if len(groups) < 2:
            continue
        say(f"  per {key}:")
        for g in groups:
            mk = np.isin(rec_of, [rt for rt in recs if meta[rt].get(key, "?") == g])
            say(f"    {g:12s} n={mk.sum():4d}  " + "  ".join(f"{100 * finger_ok[m][mk].mean():5.1f}" for m in show))

    # ---- 4. error analysis by subset
    pitch = np.concatenate([ref[rt]["pitch"][sel[rt]] for rt in recs])
    onset = np.concatenate([ref[rt]["onset"][sel[rt]] for rt in recs])
    auto = np.concatenate([ref[rt]["auto"][sel[rt]] for rt in recs])
    pres = (np.concatenate([ref[rt]["present"][sel[rt]] for rt in recs])
            if all("present" in ref[rt] for rt in recs) else None)
    chord = np.zeros(n, bool)
    ioi = np.full(n, 9.0)
    off = 0
    for rt in recs:
        d = ref[rt]
        nts = notes_of(d)
        ch = np.zeros(len(nts), bool)
        for g in H.chord_groups(nts, 0.034):
            if len(g) > 1:
                ch[g] = True
        on_all = d["onset"]
        gap = np.minimum(np.r_[9.0, np.diff(on_all)], np.r_[np.diff(on_all), 9.0])
        k = len(sel[rt])
        chord[off:off + k] = ch[sel[rt]]
        ioi[off:off + k] = gap[sel[rt]]
        off += k
    black = np.isin(pitch % 12, (1, 3, 6, 8, 10))
    subsets = [(f"true finger {f}", (y % 5) == f - 1) for f in range(1, 6)]
    subsets += [("white key", ~black), ("black key", black), ("in a chord", chord), ("single note", ~chord),
                ("fast (IOI < 0.1 s)", ioi < 0.1), ("left hand", y < 5), ("right hand", y >= 5),
                ("against C4 split", ((y < 5) & (pitch >= 60)) | ((y >= 5) & (pitch < 60))),
                ("low register (< C3)", pitch < 48), ("high register (> C6)", pitch > 84)]
    if args.labels == "manual":
        subsets += [("dataset labels it", auto >= 0), ("dataset Noinfo", auto < 0)]
    if pres is not None:
        subsets += [("both hands seen", pres.all(1)), ("one hand seen", pres.sum(1) == 1),
                    ("no hand seen", ~pres.any(1))]
    sub_show = [main_m] + [m for m in ("roi", "ayf", "auto") if m in res]
    other = [m for m in methods if ":" in m and m != main_m and m.split(":")[0] != main_m.split(":")[0]]
    sub_show += [m for m in methods if ":" in m and m != main_m and m.split(":")[0] == main_m.split(":")[0]][:2]
    sub_show += [m for m in other if m.endswith(":seq")][:3]
    sub_show = list(dict.fromkeys(sub_show))
    say(f"\n  by subset, hand + finger % ({' | '.join(sub_show)})")
    for name, mk in subsets:
        if mk.sum() == 0:
            continue
        say(f"    {name:22s} n={mk.sum():5d}  " + "  ".join(f"{100 * finger_ok[m][mk].mean():5.1f}" for m in sub_show))

    # ---- 5. error types
    say("\n  error types (% of all notes):")
    for m in [main_m] + [x for x in ("roi", "ayf") if x in res]:
        cls = res[m][0][0]
        no = cls < 0
        wh = (~no) & (cls // 5 != y // 5)
        fo = (~no) & (~wh) & (cls != y)
        dist = np.abs(cls % 5 - y % 5)
        say(f"    {m:24s} no decision {100 * no.mean():4.1f} | wrong hand {100 * wh.mean():4.1f} | wrong finger "
            f"{100 * fo.mean():4.1f} (off by 1: {100 * (fo & (dist == 1)).mean():4.1f}, by 2+: "
            f"{100 * (fo & (dist >= 2)).mean():4.1f})")

    # ---- 6. precision-coverage
    say("\n  precision (hand + finger %) when answering only the most confident notes, by share answered:")
    say(f"    {'method':24s} " + " ".join(f"{c:>5d}%" for c in COVERAGES))
    curves = {}
    for m in [x for x in methods if x not in ("pitch", "auto")]:
        cls, _, cc = res[m][0]
        order = np.argsort(-cc, kind="stable")
        corr = ((cls == y) & (cls >= 0))[order].astype(float)
        prec = np.cumsum(corr) / np.arange(1, n + 1)
        curves[m] = prec
        say(f"    {m:24s} " + " ".join(f"{100 * prec[max(0, int(round(c / 100 * n)) - 1)]:6.1f}" for c in COVERAGES))
    if "auto" in res:
        a = res["auto"][0][0]
        cov = (a >= 0).mean()
        prec = ((a == y) & (a >= 0)).sum() / max(1, (a >= 0).sum())
        say(f"    dataset labels answer {100 * cov:.1f} % of the notes with precision {100 * prec:.1f} %; at that "
            f"share: " + ", ".join(f"{m} {100 * curves[m][max(0, int(round(cov * n)) - 1)]:.1f}"
                                   for m in curves if m in sub_show))
    if args.curves:
        cd = Path(args.curves)
        cd.mkdir(parents=True, exist_ok=True)
        with open(cd / f"precision_coverage_{args.labels}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["coverage"] + list(curves))
            for i in range(n):
                w.writerow([round((i + 1) / n, 5)] + [round(float(curves[m][i]), 5) for m in curves])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 4))
            for m in [x for x in sub_show if x in curves]:
                ax.plot(100 * np.arange(1, n + 1) / n, 100 * curves[m], label=m)
            if "auto" in res:
                ax.plot([100 * cov], [100 * prec], "k*", ms=10, label="dataset labels")
            ax.set_xlabel("notes answered (%)")
            ax.set_ylabel("hand + finger precision (%)")
            ax.set_xlim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(cd / f"precision_coverage_{args.labels}.png", dpi=150)
            plt.close(fig)
        except Exception as e:                      # noqa: BLE001
            say(f"  (no plot: {e})")

    # ---- 7. finger confusion of the main method
    cls = res[main_m][0][0]
    say(f"\n  confusion of {main_m} (rows true, columns predicted; '-' = no decision):")
    say("         " + " ".join(f"{c:>4s}" for c in H.CLASS_NAMES) + "    -")
    for t in range(H.N_CLASSES):
        mk = y == t
        if not mk.any():
            continue
        row = [int(((cls == p) & mk).sum()) for p in range(H.N_CLASSES)] + [int(((cls < 0) & mk).sum())]
        say(f"    {H.CLASS_NAMES[t]:4s} " + " ".join(f"{v:4d}" for v in row))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        rows_idx = np.concatenate([ref[rt]["rows"][sel[rt]] for rt in recs])
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record", "row", "onset", "pitch", "label", "dataset"] + methods)
            for i in range(n):
                w.writerow([rec_of[i], int(rows_idx[i]), round(float(onset[i]), 4), int(pitch[i]),
                            H.CLASS_NAMES[y[i]] if y[i] >= 0 else "?",
                            H.CLASS_NAMES[auto[i]] if auto[i] >= 0 else "Noinfo"]
                           + [H.CLASS_NAMES[c] if c >= 0 else "-" for c in (res[m][0][0][i] for m in methods)])
        say(f"\n  per-note decisions -> {args.csv}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
