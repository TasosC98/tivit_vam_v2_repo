"""Note-level analyses of the PianoVAM test predictions (paper 1 and its review).

Reads predicted MIDI folders (what evaluate_run.sh writes to out/<name>_test) and the
dataset labels; no GPU. Each --system is one folder per training run, comma-separated:

    python paper/analyse_notes.py --config configs/vam_full.yaml \
        --system KeyWise=out/E1_vam_full_test,out/E1s2_vam_full_test,out/E1s3_vam_full_test \
        --system Global=out/A1_vam_strip_test,out/A1s2_vam_strip_test,out/A1s3_vam_strip_test \
        --system V2N=/home/achatzigiannis/V2N/predicted_midi/pianovam_test \
        --out paper/data/note_analysis.json

On a machine without a server profile, add the dataset folder at the end, e.g.
data.root=C:/Users/User/Desktop/PHD/server/PianoVAM_labels
The offset numbers need the corrected PianoVAM labels (TSV files fixed on Hugging Face on
2026-03-18, as on the server); with an older download only the onset analyses are valid.

Notes are matched exactly as mir_eval does (maximum matching per pitch). For every system
(runs averaged per recording; per-run values are kept too):
  errors    error classes at 50 ms (Table 3): a missed reference note, or an extra predicted
            note, is "timing" if it is matched at 100 ms, "neighbour" if an unmatched note one
            or two semitones away starts within 50 ms, "octave" if one 12 semitones away does,
            and "other" otherwise;
  recall    share of reference notes found at 50 ms by register and key colour, with 95%
            bootstrap intervals over the recordings, and paired differences between systems;
  timing    signed onset error (predicted - reference) of the notes matched at 100 ms (the
            paper's numbers) and at 50 ms, by key colour and per recording;
  offsets   offset-only F1 (same key, release within 50 / 100 ms, onset ignored) and the
            release error of the notes matched on onset;
  repeats   recall of the notes that repeat a key less than 150 ms after its previous onset;
  per key   recall and median onset error of each key, the distance between the key centre
            and the centre of its tile / encoder column on a standard keyboard, the number of
            training notes of each key, and Spearman correlations between them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pianovam_vision.config import load_config  # noqa: E402
from pianovam_vision.labels import read_reference  # noqa: E402
from pianovam_vision.metadata import filter_by_split, recordings_from_cfg  # noqa: E402
from pianovam_vision.score_midi import load_midi_notes  # noqa: E402

PITCHES = np.arange(21, 109)
BLACK = {1, 3, 6, 8, 10}
REGISTERS = [("A0-B1", 21, 35), ("C2-B3", 36, 59), ("C4-B5", 60, 83), ("C6-C8", 84, 108)]
GROUPS = [g[0] for g in REGISTERS] + ["white", "black", "all"]
REPEAT_S = 0.15


def is_black(p: int) -> bool:
    return p % 12 in BLACK


def group_of(p: int):
    reg = next(name for name, lo, hi in REGISTERS if lo <= p <= hi)
    return reg, ("black" if is_black(p) else "white"), "all"


def key_geometry(width: float = 1408.0, tile: float = 16.0, column_shift: float = 7.5):
    """Key centres on a standard keyboard (52 equal white keys, each black key centred on the
    boundary of its two white neighbours) and their offsets from the tile / encoder-column
    centres, in strip pixels. Column k's receptive field is centred 7.5 px left of tile k."""
    ww = width / 52
    centres, wi = [], 0
    for p in PITCHES:
        if is_black(int(p)):
            centres.append(wi * ww)
        else:
            centres.append((wi + 0.5) * ww)
            wi += 1
    centres = np.array(centres)
    tiles = tile * np.arange(88) + tile / 2
    return centres, tiles - centres, tiles - column_shift - centres


def match(ref, est, tol, offsets=False):
    """(reference index, predicted index) pairs of mir_eval's maximum matching, per pitch:
    onset within ``tol`` (offsets=False), or release within ``tol`` ignoring the onset."""
    import mir_eval

    by_r, by_e = defaultdict(list), defaultdict(list)
    for i, n in enumerate(ref):
        by_r[n.pitch].append(i)
    for j, n in enumerate(est):
        by_e[n.pitch].append(j)
    pairs = []
    for p in by_r.keys() & by_e.keys():
        ri, ei = by_r[p], by_e[p]
        r_int = np.array([[ref[i].onset, ref[i].offset] for i in ri])
        e_int = np.array([[est[j].onset, est[j].offset] for j in ei])
        if offsets:
            m = mir_eval.transcription.match_note_offsets(r_int, e_int, offset_ratio=0.0,
                                                          offset_min_tolerance=tol)
        else:
            m = mir_eval.transcription.match_notes(r_int, np.full(len(ri), 440.0), e_int,
                                                   np.full(len(ei), 440.0), onset_tolerance=tol,
                                                   offset_ratio=None)
        pairs += [(ri[a], ei[b]) for a, b in m]
    return pairs


def classify(notes, unmatched, others, others_unmatched, matched100):
    """Error class of every unmatched note (see the module docstring)."""
    on = np.array([n.onset for n in others])
    order = np.argsort(on)
    on_sorted = on[order]
    out = defaultdict(int)
    for i in np.flatnonzero(unmatched):
        if matched100[i]:
            out["timing"] += 1
            continue
        n = notes[i]
        lo, hi = np.searchsorted(on_sorted, [n.onset - 0.05 - 1e-9, n.onset + 0.05 + 1e-9])
        near = [abs(others[j].pitch - n.pitch) for j in order[lo:hi] if others_unmatched[j]]
        if any(d in (1, 2) for d in near):
            out["neighbour"] += 1
        elif 12 in near:
            out["octave"] += 1
        else:
            out["other"] += 1
    return dict(out)


def prf(n_match, n_ref, n_est):
    p = n_match / n_est if n_est else 0.0
    r = n_match / n_ref if n_ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def analyse_recording(ref, est):
    m50, m100 = match(ref, est, 0.05), match(ref, est, 0.10)
    r50, e50 = np.zeros(len(ref), bool), np.zeros(len(est), bool)
    r100, e100 = np.zeros(len(ref), bool), np.zeros(len(est), bool)
    for i, j in m50:
        r50[i] = e50[j] = True
    for i, j in m100:
        r100[i] = e100[j] = True

    found, total = defaultdict(int), defaultdict(int)
    key_found, key_total = np.zeros(88), np.zeros(88)
    for i, n in enumerate(ref):
        for g in group_of(n.pitch):
            total[g] += 1
            found[g] += int(r50[i])
        key_total[n.pitch - 21] += 1
        key_found[n.pitch - 21] += r50[i]

    last, rep_total, rep_found = {}, 0, 0
    for i, n in enumerate(ref):                       # ref is sorted by onset
        if n.pitch in last and n.onset - last[n.pitch] < REPEAT_S:
            rep_total += 1
            rep_found += int(r50[i])
        last[n.pitch] = n.onset

    def errors(pairs):
        e = [(1000 * (est[j].onset - ref[i].onset), ref[i].pitch) for i, j in pairs]
        return np.array([x for x, _ in e]), np.array([p for _, p in e], dtype=int)

    err50, pit50 = errors(m50)
    err100, pit100 = errors(m100)
    rel = np.array([1000 * (est[j].offset - ref[i].offset) for i, j in m50])
    off = {}
    for tol in (0.05, 0.10):
        k = len(match(ref, est, tol, offsets=True))
        off[f"offset_only_f1@{int(round(tol * 1000))}"] = prf(k, len(ref), len(est))
    return {
        "n_ref": len(ref), "n_est": len(est), "matched50": len(m50), "matched100": len(m100),
        "found": dict(found), "total": dict(total),
        "key_found": key_found, "key_total": key_total,
        "repeats": [rep_total, rep_found],
        "missed": classify(ref, ~r50, est, ~e50, r100),
        "extra": classify(est, ~e50, ref, ~r50, e100),
        "err50": err50, "pit50": pit50, "err100": err100, "pit100": pit100,
        "release_err_ms": rel, **off,
    }


def bootstrap(values_fn, n_rec, n_boot, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n_rec, size=(n_boot, n_rec))
    return np.array([values_fn(ix) for ix in idx])


def spearman(x, y):
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    r = spearmanr(x[ok], y[ok])
    return {"rho": round(float(r.statistic), 3), "p": float(r.pvalue), "n": int(ok.sum())}


def median(a):
    return float(np.median(a)) if len(a) else float("nan")


def summarise_run(per_rec):
    """Pooled numbers of one run (Table 3, Fig. 4 and the text of Section 5.3)."""
    s = {"matched_50ms": int(sum(r["matched50"] for r in per_rec.values())),
         "true_notes": int(sum(r["n_ref"] for r in per_rec.values())),
         "predicted_notes": int(sum(r["n_est"] for r in per_rec.values()))}
    for k in ("missed", "extra"):
        tot = defaultdict(int)
        for r in per_rec.values():
            for c, v in r[k].items():
                tot[c] += v
        s[f"{k}_by_cause"] = dict(sorted(tot.items(), key=lambda kv: -kv[1]))
    s["recall_by_group"] = {g: [int(sum(r["found"].get(g, 0) for r in per_rec.values())),
                                int(sum(r["total"].get(g, 0) for r in per_rec.values()))] for g in GROUPS}
    s["fast_repeats_total_found"] = [int(sum(r["repeats"][k] for r in per_rec.values())) for k in (0, 1)]
    for tol in ("50", "100"):
        err = np.concatenate([r[f"err{tol}"] for r in per_rec.values()])
        pit = np.concatenate([r[f"pit{tol}"] for r in per_rec.values()])
        blk = np.array([is_black(int(p)) for p in pit], bool)
        s[f"median_onset_error_ms@{tol}"] = {"white": median(err[~blk]), "black": median(err[blk]),
                                             "all": median(err)}
        s[f"median_onset_error_ms@{tol}_per_recording"] = {k: median(r[f"err{tol}"]) for k, r in per_rec.items()}
    s["share_of_100ms_matches_outside_50ms"] = float(np.mean(np.abs(np.concatenate(
        [r["err100"] for r in per_rec.values()])) > 50.0 + 1e-6))
    rel = np.concatenate([r["release_err_ms"] for r in per_rec.values()])
    s["release_error_ms"] = {"median": median(rel), "median_abs": median(np.abs(rel))}
    for tol in ("50", "100"):
        s[f"offset_only_f1@{tol}_mean_over_recordings"] = float(np.mean(
            [r[f"offset_only_f1@{tol}"] for r in per_rec.values()]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--system", action="append", required=True,
                    help="NAME=folder[,folder...] (one folder of <record>.mid per run)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None, help="JSON with every number")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    excl = set(cfg["data"].get("exclude_records", []) or [])
    all_recs = [r for r in recordings_from_cfg(cfg) if r.record_time not in excl]
    recs = filter_by_split(all_recs, [args.split])
    refs = {r.record_time: read_reference(r, cfg) for r in recs}
    names = sorted(refs)

    systems = {}
    for spec in args.system:
        name, folders = spec.split("=", 1)
        systems[name] = [f for f in folders.split(",") if f]

    # --- every run of every system, per recording
    res = {}
    for name, folders in systems.items():
        res[name] = []
        for f in folders:
            per = {}
            for k in names:
                path = os.path.join(f, f"{k}.mid")
                if not os.path.exists(path):
                    raise SystemExit(f"missing {path}")
                per[k] = analyse_recording(refs[k], load_midi_notes(path))
            res[name].append(per)
            print(f"{name}: {f} done")

    out = {"about": __doc__.strip().splitlines()[0], "split": args.split, "recordings": names,
           "systems": {}, "differences": {}, "per_key": {}}
    centres, dt, dc = key_geometry()
    out["per_key"]["geometry"] = {"pitch": PITCHES.tolist(), "key_centre_px": centres.round(2).tolist(),
                                  "tile_minus_key_px": dt.round(2).tolist(),
                                  "column_minus_key_px": dc.round(2).tolist()}

    # training notes per key (the splits the models were trained on)
    train = filter_by_split(all_recs, cfg["data"]["train_splits"])
    train_count = np.zeros(88)
    for r in train:
        for n in read_reference(r, cfg):
            if 21 <= n.pitch <= 108:
                train_count[n.pitch - 21] += 1
    out["per_key"]["training_notes"] = train_count.astype(int).tolist()
    out["per_key"]["test_notes"] = sum(r["key_total"] for r in res[next(iter(res))][0].values()).astype(int).tolist()

    # per recording found/total, runs averaged: (systems) x (recordings) x (groups)
    F = {n: np.array([[np.mean([run[k]["found"].get(g, 0) for run in res[n]]) for g in GROUPS] for k in names])
         for n in res}
    T = np.array([[res[next(iter(res))][0][k]["total"].get(g, 0) for g in GROUPS] for k in names], float)
    boot_idx = np.random.default_rng(0).integers(0, len(names), size=(args.bootstrap, len(names)))

    def recall_ci(Fn):
        b = np.array([Fn[ix].sum(0) / T[ix].sum(0) for ix in boot_idx])
        return np.nanpercentile(b, [2.5, 97.5], axis=0)

    for n, runs in res.items():
        per_run = [summarise_run(run) for run in runs]
        rec = F[n].sum(0) / T.sum(0)
        lo, hi = recall_ci(F[n])
        key_found = np.mean([sum(r["key_found"] for r in run.values()) for run in runs], axis=0)
        key_total = sum(r["key_total"] for r in runs[0].values())
        key_err = [median(np.concatenate([np.concatenate([r["err100"][r["pit100"] == p] for r in run.values()])
                                          for run in runs])) for p in PITCHES]
        key_recall = np.where(key_total > 0, key_found / np.maximum(key_total, 1), np.nan)
        out["systems"][n] = {
            "runs": systems[n],
            "recall@50": {g: {"mean_over_runs": float(rec[i]), "ci95": [float(lo[i]), float(hi[i])]}
                          for i, g in enumerate(GROUPS)},
            "per_run": per_run,
        }
        out["per_key"][n] = {"recall@50": [None if np.isnan(x) else round(float(x), 4) for x in key_recall],
                             "median_onset_error_ms@100": [None if np.isnan(x) else round(x, 2) for x in key_err]}
        enough = key_total >= 20
        kr = np.where(enough, key_recall, np.nan)
        ke = np.where(enough, np.array(key_err), np.nan)
        blk = np.array([is_black(int(p)) for p in PITCHES])
        cor = {}
        for label, sel in (("all", np.ones(88, bool)), ("white", ~blk), ("black", blk)):
            x = np.where(sel, kr, np.nan)
            e = np.where(sel, ke, np.nan)
            cor[label] = {"recall_vs_abs_column_offset": spearman(np.abs(dc), x),
                          "recall_vs_abs_tile_offset": spearman(np.abs(dt), x),
                          "onset_error_vs_column_offset": spearman(dc, e),
                          "recall_vs_log_training_notes": spearman(np.log1p(train_count), x)}
        out["per_key"][n]["spearman_keys_with_20_test_notes"] = cor

    # paired differences between systems (same bootstrap resamples)
    names_sys = list(res)
    for a_i, a in enumerate(names_sys):
        for b in names_sys[a_i + 1:]:
            d = F[a].sum(0) / T.sum(0) - F[b].sum(0) / T.sum(0)
            bd = np.array([F[a][ix].sum(0) / T[ix].sum(0) - F[b][ix].sum(0) / T[ix].sum(0) for ix in boot_idx])
            lo, hi = np.nanpercentile(bd, [2.5, 97.5], axis=0)
            ka = np.array([np.nan if v is None else v for v in out["per_key"][a]["recall@50"]], float)
            kb = np.array([np.nan if v is None else v for v in out["per_key"][b]["recall@50"]], float)
            key_total = np.array(out["per_key"]["test_notes"])
            gain = np.where(key_total >= 20, ka - kb, np.nan)
            out["differences"][f"{a}-{b}"] = {
                "recall@50": {g: {"diff": float(d[i]), "ci95": [float(lo[i]), float(hi[i])]}
                              for i, g in enumerate(GROUPS)},
                "per_key_gain_vs_log_training_notes": spearman(np.log1p(train_count), gain),
            }

    # --- printout
    for n, s in out["systems"].items():
        print(f"\n=== {n} ({len(systems[n])} run(s))")
        print("recall@50 (runs averaged), 95% bootstrap CI over recordings:")
        for g in GROUPS:
            r = s["recall@50"][g]
            print(f"  {g:6s} {100 * r['mean_over_runs']:5.1f}  [{100 * r['ci95'][0]:5.1f}, {100 * r['ci95'][1]:5.1f}]")
        for i, pr in enumerate(s["per_run"]):
            print(f"  run {i + 1}: matched {pr['matched_50ms']}/{pr['true_notes']} (pred {pr['predicted_notes']}),"
                  f" missed {pr['missed_by_cause']}, extra {pr['extra_by_cause']}")
            print(f"         repeats (total, found) {pr['fast_repeats_total_found']}, median onset error @100"
                  f" white {pr['median_onset_error_ms@100']['white']:+.1f} black {pr['median_onset_error_ms@100']['black']:+.1f} ms,"
                  f" offset-only F1 @50/@100 {100 * pr['offset_only_f1@50_mean_over_recordings']:.1f}/"
                  f"{100 * pr['offset_only_f1@100_mean_over_recordings']:.1f}")
            pr_rec = pr["median_onset_error_ms@100_per_recording"]
            print(f"         per-recording median onset error @100: {min(pr_rec.values()):+.1f} .. {max(pr_rec.values()):+.1f} ms;"
                  f" 100-ms matches outside 50 ms: {100 * pr['share_of_100ms_matches_outside_50ms']:.1f}%")
        for label, c in out["per_key"][n]["spearman_keys_with_20_test_notes"].items():
            print(f"  per-key Spearman ({label}): " + "; ".join(
                f"{k} rho={v['rho']:+.2f} p={v['p']:.3f} n={v['n']}" for k, v in c.items()))
    for k, d in out["differences"].items():
        print(f"\n=== {k}: recall@50 difference, 95% bootstrap CI")
        for g in GROUPS:
            r = d["recall@50"][g]
            print(f"  {g:6s} {100 * r['diff']:+5.1f}  [{100 * r['ci95'][0]:+5.1f}, {100 * r['ci95'][1]:+5.1f}]")
        v = d["per_key_gain_vs_log_training_notes"]
        print(f"  per-key gain vs log(training notes): rho={v['rho']:+.2f} p={v['p']:.3f} n={v['n']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        print(f"\nall numbers -> {args.out}")


if __name__ == "__main__":
    main()
