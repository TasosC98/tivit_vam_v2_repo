"""Figures 3 and 4 of the paper, made from the evaluation results.

    python paper/make_figures.py --results results --analysis results/note_analysis.json \
        --keywise pianovam_s1,pianovam_s2,pianovam_s3 --global global_s1,global_s2,global_s3

Writes <out>/Fig3.pdf and Fig4.pdf (and .png). Needs numpy and matplotlib. Both are drawn
at their printed size, so their lettering is not scaled down by LaTeX. Data used:
  <results>/<name>_vs_baseline.csv  per-recording scores of each run and of V2N
                                    (scripts/evaluate.sh with BASELINE_MIDI)
  --analysis                        paper/analyse_notes.py output with the systems
                                    KeyWise, Global and (optionally) Global88
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "Arial"})
C_KEY, C_GLOBAL, C_GLOBAL88, C_V2N = "#0B7D74", "#8A9A97", "#C9D1CF", "#2F5FD0"
NL = "\n"


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def read_csv(path):
    with open(path, newline="") as f:
        return {r["record"]: r for r in csv.DictReader(f)}


def fig3_recordings(results, keywise, global_runs, out):
    """Per recording: KeyWise (mean of the runs, bar = min..max), global variant (mean of its runs), V2N."""
    runs = [read_csv(os.path.join(results, f"{n}_vs_baseline.csv")) for n in keywise]
    key = runs[0]
    glos = [read_csv(os.path.join(results, f"{n}_vs_baseline.csv")) for n in global_runs]
    glo_label = "Global variant (3.7 M)" + (f", mean of {len(glos)} runs" if len(glos) > 1 else "")
    mean = lambda r, tol: sum(100 * float(k[r][f"pred:onset_f1@{tol}"]) for k in runs) / len(runs)
    recs = sorted(key, key=lambda r: mean(r, "50"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.75), sharey=True)
    for ax, tol, part in zip(axes, ("50", "100"), ("a", "b")):
        y = list(range(len(recs)))
        vals = [[100 * float(k[r][f"pred:onset_f1@{tol}"]) for k in runs] for r in recs]
        ax.hlines(y, [min(v) for v in vals], [max(v) for v in vals], color=C_KEY, lw=1.2, alpha=0.55, zorder=2)
        ax.scatter([sum(100 * float(g[r][f"pred:onset_f1@{tol}"]) for g in glos) / len(glos) for r in recs], y,
                   marker="s", s=14, color=C_GLOBAL, label=glo_label, zorder=3)
        ax.scatter([100 * float(key[r][f"baseline:onset_f1@{tol}"]) for r in recs], y, marker="^", s=16, color=C_V2N, label="V2N (28.2 M)", zorder=3)
        ax.scatter([sum(v) / len(v) for v in vals], y, marker="o", s=16, color=C_KEY,
                   label=f"KeyWise (1.3 M), mean and range of {len(runs)} runs", zorder=4)
        ax.set_xlabel(f"({part}) Onset F1 (%), {tol} ms tolerance")
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(82, 100.5)
    axes[0].set_yticks(list(range(len(recs))))
    axes[0].set_yticklabels(recs)                 # PianoVAM recording IDs
    axes[0].set_ylabel("PianoVAM test recording")
    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 0, 1]                             # KeyWise, global variant, V2N
    fig.tight_layout(rect=(0, 0, 1, 0.89), pad=0.3)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, out, "Fig3")


def fig4_register(analysis, out):
    """Recall by register / key colour, runs averaged, with 95% bootstrap intervals over the recordings."""
    groups = [("A0–B1", "A0-B1"), ("C2–B3", "C2-B3"), ("C4–B5", "C4-B5"), ("C6–C8", "C6-C8"),
              ("White" + NL + "keys", "white"), ("Black" + NL + "keys", "black")]
    bars = [("per_key", C_KEY, "KeyWise"), ("global", C_GLOBAL, "Global variant")]
    hatch = {"per_key": None, "global": "///", "global88": "..."}   # readable without colour
    with open(analysis, encoding="utf-8") as f:
        d = json.load(f)["systems"]
    src = {"per_key": "KeyWise", "global": "Global"}
    if "Global88" in d:                        # Global-88 ablation, when analysed
        src["global88"] = "Global88"
        bars.append(("global88", C_GLOBAL88, "Global-88"))
    rec = {m: [100 * d[s]["recall@50"][k]["mean_over_runs"] for _, k in groups] for m, s in src.items()}
    err = {m: [[rec[m][i] - 100 * d[s]["recall@50"][k]["ci95"][0] for i, (_, k) in enumerate(groups)],
               [100 * d[s]["recall@50"][k]["ci95"][1] - rec[m][i] for i, (_, k) in enumerate(groups)]]
           for m, s in src.items()}
    fig, ax = plt.subplots(figsize=(3.2, 1.62))
    x = range(len(groups))
    width = 0.84 / len(bars)
    offsets = [(j - (len(bars) - 1) / 2) * width for j in range(len(bars))]
    for dx, (m, c, lab) in zip(offsets, bars):
        ax.bar([i + dx for i in x], rec[m], width=width * 0.95, color=c, label=lab, hatch=hatch[m],
               edgecolor="#333333", linewidth=0.3,
               yerr=err[m], error_kw=dict(lw=0.6, capsize=1.2, ecolor="#333333"))
    for i in x:
        for dx, (m, _, _) in zip(offsets, bars):
            top = rec[m][i] + err[m][1][i]
            ax.text(i + dx, top + 0.4, f"{rec[m][i]:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(60, 106)
    ax.set_ylabel("Notes found, 50 ms (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(3.5, color="#999999", lw=0.6, ls="--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=len(bars), frameon=False)
    fig.tight_layout(pad=0.3)
    save(fig, out, "Fig4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with <name>_vs_baseline.csv")
    ap.add_argument("--analysis", required=True, help="note_analysis.json from paper/analyse_notes.py")
    ap.add_argument("--keywise", default="pianovam_s1,pianovam_s2,pianovam_s3",
                    help="evaluation names of the KeyWise runs (comma-separated)")
    ap.add_argument("--global", dest="global_runs", default="global_s1,global_s2,global_s3",
                    help="evaluation names of the global-variant runs (comma-separated)")
    ap.add_argument("--out", default="figures", help="output folder")
    a = ap.parse_args()
    fig3_recordings(a.results, a.keywise.split(","), a.global_runs.split(","), a.out)
    fig4_register(a.analysis, a.out)
    print("figures written to", a.out)
