"""Figures of the paper draft, made from the evaluation result files.

    python paper/make_figures.py --results C:/Users/User/Desktop/PHD/server/results

Writes figures/fig1_model.pdf, fig2_recordings.pdf, fig3_register.pdf (for LaTeX)
and the same names as .png (for the HTML report). Needs numpy and matplotlib.
Data used:
  <results>/E1_vam_full_vs_baseline.csv   per-key model vs V2N (score_midi)
  <results>/A1_vam_strip_vs_baseline.csv  global-strip model vs V2N (score_midi)
  paper/data/error_analysis.json          recall by register / key colour
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
C_KEY, C_GLOBAL, C_GLOBAL88, C_V2N = "#0B7D74", "#8A9A97", "#C9D1CF", "#2F5FD0"
NL = "\n"


def save(fig, name):
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig1_model():
    """Pipeline: frame -> warp -> strip -> CNN -> 88 columns -> shared BiGRU -> outputs."""
    fig, ax = plt.subplots(figsize=(7.0, 1.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 23)
    ax.axis("off")

    def box(x, w, title, lines):
        ax.add_patch(Rectangle((x, 1), w, 21, fc="#F3F5F7", ec="#333333", lw=0.8))
        ax.text(x + w / 2, 20.2, title, ha="center", va="top", fontsize=6.3, fontweight="bold")
        ax.text(x + w / 2, 2.4, NL.join(lines), ha="center", va="bottom", fontsize=5.6, linespacing=1.3)

    def arrow(x0, x1, y=11.5):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333"))

    ws = [12.5, 19.0, 15.0, 15.0, 15.0, 15.0]
    xs = [sum(ws[:i]) + 1.7 * i for i in range(6)]
    box(xs[0], ws[0], "Video frame", ["keyboard seen", "from above", "720p, 30 fps"])
    box(xs[1], ws[1], "Rectified strip", ["1408 × 112 px,", "88 columns of 16 px"])
    for i in range(88):                       # the 88 key columns, middle C highlighted
        ax.add_patch(Rectangle((xs[1] + 1.0 + i * (ws[1] - 2.0) / 88, 10.0), (ws[1] - 2.0) / 88, 6.6,
                               fc=C_KEY if i == 39 else "white", ec="#9AA3AE", lw=0.15))
    box(xs[2], ws[2], "CNN encoder", ["4 blocks,", "32/64/128/192 ch,", "stride 2", "→ 88 × 7 × 192"])
    box(xs[3], ws[3], "Column pooling", ["mean of each", "column, shared", "1×1 projection", "→ 88 × 64"])
    box(xs[4], ws[4], "Shared BiGRU", ["2 layers,", "128 units/direction,", "same weights", "for all 88 keys"])
    box(xs[5], ws[5], "Two heads", ["onset and frame", "probability for", "each key and", "video frame"])
    for i in range(5):
        arrow(xs[i] + ws[i] + 0.2, xs[i + 1] - 0.2)
    save(fig, "fig1_model")


def read_csv(path):
    with open(path, newline="") as f:
        return {r["record"]: r for r in csv.DictReader(f)}


def fig2_recordings(results):
    """Per recording: per-key network (mean of the three runs, bar = min..max), global variant, V2N."""
    runs = [read_csv(os.path.join(results, f"{n}_vam_full_vs_baseline.csv")) for n in ("E1", "E1s2", "E1s3")]
    key = runs[0]
    glos = [read_csv(p) for p in (os.path.join(results, f"{n}_vam_strip_vs_baseline.csv") for n in ("A1", "A1s2", "A1s3"))
            if os.path.exists(p)]
    glo_label = "Global variant (3.7 M)" + (f", mean of {len(glos)} runs" if len(glos) > 1 else "")
    mean = lambda r, tol: sum(100 * float(k[r][f"pred:onset_f1@{tol}"]) for k in runs) / len(runs)
    recs = sorted(key, key=lambda r: mean(r, "50"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, tol in zip(axes, ("50", "100")):
        y = list(range(len(recs)))
        vals = [[100 * float(k[r][f"pred:onset_f1@{tol}"]) for k in runs] for r in recs]
        ax.hlines(y, [min(v) for v in vals], [max(v) for v in vals], color=C_KEY, lw=1.2, alpha=0.55, zorder=2)
        ax.scatter([sum(100 * float(g[r][f"pred:onset_f1@{tol}"]) for g in glos) / len(glos) for r in recs], y,
                   marker="s", s=14, color=C_GLOBAL, label=glo_label, zorder=3)
        ax.scatter([100 * float(key[r][f"baseline:onset_f1@{tol}"]) for r in recs], y, marker="^", s=16, color=C_V2N, label="V2N (28.2 M)", zorder=3)
        ax.scatter([sum(v) / len(v) for v in vals], y, marker="o", s=16, color=C_KEY, label="KeyWise (1.3 M), mean and range of 3 runs", zorder=4)
        ax.set_xlabel(f"Onset F1 (%), {tol} ms tolerance")
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(82, 100.5)
    axes[0].set_yticks(list(range(len(recs))))
    axes[0].set_yticklabels(recs)                 # PianoVAM recording IDs
    axes[0].set_ylabel("PianoVAM test recording")
    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 0, 1]                             # KeyWise, global variant, V2N
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, "fig2_recordings")


def fig3_register(analysis=None):
    """Recall by register / key colour. With ``analysis`` (paper/analyse_notes.py output with
    systems KeyWise and Global): runs averaged, with 95% bootstrap intervals over recordings.
    Otherwise the first-run numbers of data/error_analysis.json."""
    groups = [("A0–B1", "recall_by_register", "A0-B1"), ("C2–B3", "recall_by_register", "C2-B3"),
              ("C4–B5", "recall_by_register", "C4-B5"), ("C6–C8", "recall_by_register", "C6-C8"),
              ("White" + NL + "keys", "recall_by_colour", "white"), ("Black" + NL + "keys", "recall_by_colour", "black")]
    err = None
    bars = [("per_key", C_KEY, "KeyWise"), ("global", C_GLOBAL, "Global variant")]
    if analysis:
        d = json.load(open(analysis))["systems"]
        src = {"per_key": "KeyWise", "global": "Global"}
        if "Global88" in d:                    # Global-88 ablation, when analysed
            src["global88"] = "Global88"
            bars.append(("global88", C_GLOBAL88, "Global-88"))
        rec = {m: [100 * d[s]["recall@50"][k]["mean_over_runs"] for _, _, k in groups] for m, s in src.items()}
        err = {m: [[rec[m][i] - 100 * d[s]["recall@50"][k]["ci95"][0] for i, (_, _, k) in enumerate(groups)],
                   [100 * d[s]["recall@50"][k]["ci95"][1] - rec[m][i] for i, (_, _, k) in enumerate(groups)]]
               for m, s in src.items()}
    else:
        d = json.load(open(os.path.join(HERE, "data", "error_analysis.json")))
        rec = {m: [100 * d[m][s][k][0] / d[m][s][k][1] for _, s, k in groups] for m in ("per_key", "global")}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = range(len(groups))
    width = 0.72 / len(bars)
    offsets = [(j - (len(bars) - 1) / 2) * width for j in range(len(bars))]
    for dx, (m, c, lab) in zip(offsets, bars):
        ax.bar([i + dx for i in x], rec[m], width=width * 0.95, color=c, label=lab,
               yerr=err[m] if err else None, error_kw=dict(lw=0.6, capsize=1.2, ecolor="#333333"))
    for i in x:
        for dx, (m, _, _) in zip(offsets, bars):
            top = rec[m][i] + (err[m][1][i] if err else 0.0)
            ax.text(i + dx, top + 0.4, f"{rec[m][i]:.0f}", ha="center", va="bottom",
                    fontsize=5.8 if len(bars) == 2 else 4.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(60, 101)
    ax.set_ylabel("Notes found, 50 ms (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(3.5, color="#999999", lw=0.6, ls="--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(bars), frameon=False)
    fig.tight_layout()
    save(fig, "fig3_register")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with the downloaded results/*.csv")
    ap.add_argument("--analysis", default=None,
                    help="note_analysis.json from paper/analyse_notes.py: Fig. 4 with all runs and 95%% intervals")
    a = ap.parse_args()
    fig1_model()
    fig2_recordings(a.results)
    fig3_register(a.analysis)
    print("figures written to", os.path.join(HERE, "figures"))
