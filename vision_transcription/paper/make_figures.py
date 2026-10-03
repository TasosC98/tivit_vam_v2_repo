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
C_KEY, C_GLOBAL, C_V2N = "#0B7D74", "#8A9A97", "#2F5FD0"
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
    key = read_csv(os.path.join(results, "E1_vam_full_vs_baseline.csv"))
    glo = read_csv(os.path.join(results, "A1_vam_strip_vs_baseline.csv"))
    recs = sorted(key, key=lambda r: float(key[r]["pred:onset_f1@50"]))
    label = lambda r: r[5:10] + " " + r[11:16].replace("-", ":")   # 2024-02-14_19-55-17 -> 02-14 19:55
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, tol in zip(axes, ("50", "100")):
        y = range(len(recs))
        ax.scatter([100 * float(glo[r][f"pred:onset_f1@{tol}"]) for r in recs], y, marker="s", s=14, color=C_GLOBAL, label="Global strip (3.7 M)", zorder=3)
        ax.scatter([100 * float(key[r][f"baseline:onset_f1@{tol}"]) for r in recs], y, marker="^", s=16, color=C_V2N, label="V2N (28.2 M)", zorder=3)
        ax.scatter([100 * float(key[r][f"pred:onset_f1@{tol}"]) for r in recs], y, marker="o", s=16, color=C_KEY, label="Per-key (1.3 M)", zorder=4)
        ax.set_xlabel(f"Onset F1 (%), {tol} ms tolerance")
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(84, 100.5)
    axes[0].set_yticks(list(range(len(recs))))
    axes[0].set_yticklabels([label(r) for r in recs])
    axes[0].set_ylabel("PianoVAM test recording")
    axes[1].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig2_recordings")


def fig3_register():
    d = json.load(open(os.path.join(HERE, "data", "error_analysis.json")))
    groups = [("A0–B1", "recall_by_register", "A0-B1"), ("C2–B3", "recall_by_register", "C2-B3"),
              ("C4–B5", "recall_by_register", "C4-B5"), ("C6–C8", "recall_by_register", "C6-C8"),
              ("White" + NL + "keys", "recall_by_colour", "white"), ("Black" + NL + "keys", "recall_by_colour", "black")]
    rec = {m: [100 * d[m][s][k][0] / d[m][s][k][1] for _, s, k in groups] for m in ("per_key", "global")}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = range(len(groups))
    ax.bar([i - 0.19 for i in x], rec["per_key"], width=0.36, color=C_KEY, label="Per-key")
    ax.bar([i + 0.19 for i in x], rec["global"], width=0.36, color=C_GLOBAL, label="Global strip")
    for i in x:
        for dx, m in ((-0.19, "per_key"), (0.19, "global")):
            ax.text(i + dx, rec[m][i] + 0.4, f"{rec[m][i]:.0f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(60, 101)
    ax.set_ylabel("Notes found, 50 ms (%)")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(3.5, color="#999999", lw=0.6, ls="--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig3_register")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with the downloaded results/*.csv")
    a = ap.parse_args()
    fig1_model()
    fig2_recordings(a.results)
    fig3_register()
    print("figures written to", os.path.join(HERE, "figures"))
