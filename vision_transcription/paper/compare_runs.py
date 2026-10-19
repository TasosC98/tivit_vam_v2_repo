"""Paired comparison of two systems on the PianoVAM test set, over several training runs.

Reads the per-recording files results/<name>_vs_baseline.csv that evaluate_run.sh writes
when BASELINE_MIDI is set. Each side is one or more evaluation names (one per run);
"V2N" means the baseline columns (V2N's released predictions) of the first file of --a.

    python paper/compare_runs.py --results results \
        --a E1_vam_full,E1s2_vam_full,E1s3_vam_full --b V2N
    python paper/compare_runs.py --results results \
        --a E1_vam_full,E1s2_vam_full,E1s3_vam_full --b A1_vam_strip,A1s2_vam_strip,A1s3_vam_strip

For each metric: mean +- std over runs (each run averaged over the recordings), the scores of
each run, the paired exact two-sided Wilcoxon test over the recordings on the run means
(as in the paper), the Holm-Bonferroni correction over the metrics compared, a 95% bootstrap
interval of the mean difference (10,000 resamplings of the recordings) and the number of
recordings on which --a is better. With the same number of runs on both sides, run k of
--a is also tested against run k of --b.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["onset_f1@50", "onset_f1@100", "offtol_f1@50", "offtol_f1@100", "frame_f1"]
VELOCITY = ["vel_f1@50", "vel_f1@100", "offvel_f1@50", "offvel_f1@100"]   # V2N ++Vel, ++Off++Vel


def read(results, name):
    path = os.path.join(results, f"{name}_vs_baseline.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["record"]: r for r in csv.DictReader(f)}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = running
    return adj


def p_exact(a, b):
    return float(wilcoxon(a, b, method="exact").pvalue) if np.any(a != b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with <name>_vs_baseline.csv")
    ap.add_argument("--a", required=True, help="comma-separated evaluation names (one per run)")
    ap.add_argument("--b", required=True, help="the same, or V2N")
    ap.add_argument("--metrics", default="main",
                    help="main (the five of Table 2), velocity (++Vel, ++Off++Vel at 50/100 ms; a separate"
                         " Holm family), or a comma-separated list of CSV metric names")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--json", default=None, help="also write every number here")
    args = ap.parse_args()
    metrics = {"main": METRICS, "velocity": VELOCITY}.get(args.metrics) or args.metrics.split(",")

    A = [read(args.results, n) for n in args.a.split(",")]
    if args.b == "V2N":
        B, col_b = [A[0]], "baseline:"
    else:
        B, col_b = [read(args.results, n) for n in args.b.split(",")], "pred:"
    recs = sorted(A[0])
    for t in A + B:
        if sorted(t) != recs:
            raise SystemExit("the files do not cover the same recordings")

    def scores(tables, col, m):          # (runs, recordings)
        return np.array([[float(t[r][col + m]) for r in recs] for t in tables])

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(recs), size=(args.bootstrap, len(recs)))
    rows, ps = [], []
    for m in metrics:
        a, b = scores(A, "pred:", m), scores(B, col_b, m)
        am, bm = a.mean(0), b.mean(0)
        d = 100 * (am - bm)
        boot = d[idx].mean(1)
        row = {"metric": m,
               "a_mean": 100 * a.mean(1).mean(), "a_std": 100 * a.mean(1).std(ddof=1) if len(a) > 1 else 0.0,
               "a_runs": (100 * a.mean(1)).round(2).tolist(),
               "b_mean": 100 * b.mean(1).mean(), "b_std": 100 * b.mean(1).std(ddof=1) if len(b) > 1 else 0.0,
               "b_runs": (100 * b.mean(1)).round(2).tolist(),
               "p": p_exact(am, bm), "diff": float(d.mean()),
               "ci95": np.percentile(boot, [2.5, 97.5]).round(2).tolist(),
               "a_better_on": int((am > bm).sum()), "recordings": len(recs)}
        if len(A) == len(B) and len(A) > 1:
            row["p_per_run"] = [round(p_exact(a[k], b[k]), 4) for k in range(len(A))]
        elif len(B) == 1 and len(A) > 1:
            row["p_per_run"] = [round(p_exact(a[k], b[0]), 4) for k in range(len(A))]
        rows.append(row)
        ps.append(row["p"])
    for row, q in zip(rows, holm(np.array(ps))):
        row["p_holm"] = float(q)

    print(f"A = {args.a}\nB = {args.b}\n{len(recs)} recordings; exact two-sided Wilcoxon on the run means,"
          f" Holm over {len(metrics)} metrics, 95% bootstrap CI of the mean difference\n")
    print(f"{'metric':14s} {'A':>12s} {'B':>12s} {'A-B':>6s} {'CI95':>15s} {'p':>7s} {'p_Holm':>7s} {'A better':>9s}  per-run p")
    for r in rows:
        print(f"{r['metric']:14s} {r['a_mean']:6.1f}+-{r['a_std']:3.1f} {r['b_mean']:6.1f}+-{r['b_std']:3.1f}"
              f" {r['diff']:+6.2f} [{r['ci95'][0]:+5.2f},{r['ci95'][1]:+5.2f}] {r['p']:7.4f} {r['p_holm']:7.4f}"
              f" {r['a_better_on']:>4d}/{r['recordings']}  {r.get('p_per_run', '')}")
    print("\nruns of A:", {r["metric"]: r["a_runs"] for r in rows})
    print("runs of B:", {r["metric"]: r["b_runs"] for r in rows})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"a": args.a, "b": args.b, "recordings": recs, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
