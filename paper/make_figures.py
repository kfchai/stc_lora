"""Publication figures for the STC-LoRA paper. Reads outputs/*.json artifacts.

Design (dataviz method): baselines recede into ordered grays with distinct
markers; the two STC variants carry the only hues (blue #2a78d6 / violet
#4a3aa7 -- validated pair: CVD dE 16.6, contrast >=3:1). Identity is never
color-alone: every series is direct-labeled and gets its own marker.

Run from repo root:  python paper/make_figures.py
Outputs: paper/latex/figs/*.pdf (for LaTeX) + *.png (preview).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "latex" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# palette (validated)
BLUE, VIOLET = "#2a78d6", "#4a3aa7"          # slow_stc, stc_frozen
G1, G2, G3 = "#b5b4b0", "#8a8985", "#52514e" # naive, ewc, er (light->dark)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e8e8e6"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42,
})

STYLE = {  # method -> (label, color, marker)
    "naive":      ("naive LoRA", G1, "o"),
    "ewc":        ("EWC",        G2, "s"),
    "er":         ("ER",         G3, "D"),
    "stc_frozen": ("STC-LoRA (frozen)", VIOLET, "^"),
    "slow_stc":   ("STC-LoRA (slow)",   BLUE,   "v"),
}


def agg(path):
    d = json.loads((ROOT / path).read_text())
    return d["aggregate"]


def deframe(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---- Figure 1: forgetting vs scale ------------------------------------
def fig_scaling():
    scales = ["0.5B", "1.5B", "7B"]
    files = ["outputs/p3_Qwen2.5-0.5B.json", "outputs/p3_Qwen2.5-1.5B.json",
             "outputs/p3_Qwen2.5-7B.json"]
    A = [agg(f) for f in files]
    x = range(3)

    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
    nudge = {"slow_stc": 6, "stc_frozen": -7, "ewc": -3, "er": 3}
    for m, (label, color, marker) in STYLE.items():
        y = [a[m]["forget_mean"] for a in A]
        e = [a[m]["forget_std"] for a in A]
        ax.errorbar(x, y, yerr=e, color=color, marker=marker, markersize=5,
                    linewidth=2, capsize=2.5, capthick=1, elinewidth=1)
        ax.annotate(label, (2, y[2]), xytext=(6, nudge.get(m, 0)),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=color if color in (BLUE, VIOLET) else MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(list(x), scales)
    ax.set_xlim(-0.15, 2.95)
    ax.set_xlabel("model scale (Qwen2.5)")
    ax.set_ylabel("forgetting (% ppl increase)")
    deframe(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"scaling.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 2: capacity-fair control ----------------------------------
def fig_capacity():
    d = json.loads((ROOT / "outputs/capacity_fair.json").read_text())
    er_x = [d[k]["n_trainable"] / 1e6 for k in ("er_rank8", "er_rank64", "er_rank333")]
    er_y = [d[k]["summary"]["forgetting_pct"] for k in ("er_rank8", "er_rank64", "er_rank333")]
    stc_y = d["slow_stc_r0.05"]["summary"]["forgetting_pct"]

    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    ax.plot(er_x, er_y, color=G3, marker="D", markersize=5, linewidth=2)
    ax.annotate("ER (rank 8 / 64 / 333)", (er_x[0], er_y[0]), xytext=(-2, 14),
                textcoords="offset points", ha="left", fontsize=7.5, color=MUTED)
    ax.plot([er_x[2]], [stc_y], color=BLUE, marker="v", markersize=8,
            linestyle="none")
    ax.annotate("STC-LoRA (slow)", (er_x[2], stc_y), xytext=(-8, 4),
                textcoords="offset points", ha="right", va="center",
                fontsize=7.5, color=BLUE)
    # the equal-capacity comparison
    ax.annotate("", xy=(er_x[2], stc_y + 0.8), xytext=(er_x[2], er_y[2] - 0.8),
                arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=0.8))
    ax.annotate("5x less\nat equal params", (er_x[2], (stc_y + er_y[2]) / 2),
                fontsize=7, color=MUTED, va="center", ha="right", xytext=(-10, 0),
                textcoords="offset points")
    ax.set_xscale("log")
    ax.set_ylim(bottom=2.5)
    ax.set_xlabel("trainable parameters (millions, log)")
    ax.set_ylabel("forgetting (% ppl increase)")
    deframe(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"capacity.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 3: stability-plasticity Pareto (0.5B) ----------------------
def fig_pareto():
    A = agg("outputs/p3_Qwen2.5-0.5B.json")
    joint = json.loads((ROOT / "outputs/p3_Qwen2.5-0.5B.json").read_text())["joint_ceiling"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
    for m, (label, color, marker) in STYLE.items():
        x, y = A[m]["forget_mean"], A[m]["final_mean"]
        ax.errorbar(x, y, xerr=A[m]["forget_std"], yerr=A[m]["final_std"],
                    color=color, marker=marker, markersize=6.5, linestyle="none",
                    capsize=2, elinewidth=1, capthick=1)
        dx, dy, ha = (0, 9, "center")
        if m == "slow_stc":
            dx, dy, ha = (10, -9, "left")
        elif m == "stc_frozen":
            dx, dy, ha = (10, 6, "left")
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=7.5,
                    color=color if color in (BLUE, VIOLET) else MUTED)
    ax.axhline(joint, color=MUTED, linewidth=0.8, linestyle="--")
    ax.annotate("joint ceiling", (ax.get_xlim()[1], joint), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    ax.set_xlabel("forgetting (% ppl increase)  $\\leftarrow$ better")
    ax.set_ylabel("final avg ppl  $\\leftarrow$ better")
    deframe(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"pareto.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_scaling()
    fig_capacity()
    fig_pareto()
    print("wrote", *[p.name for p in sorted(FIGS.glob('*.pdf'))])
