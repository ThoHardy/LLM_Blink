"""CoT-induced blindness figures (issue #10, simplified budget=None design).

Reads cib_master_table.csv (from analyze_cib.py) and renders:
  fig_cib_paradox.png     k=1 vs k=5 slopegraph (Direct->CoT), semantic pooled
                          -> the headline: CoT is neutral/helpful at k=1 but
                             causes blindness at k=5.
  fig_cib_by_load.png     per-load k=5 slopegraphs (trivial, semantic_1..4)
  fig_cib_vs_capacity.png CoT-induced blindness = f(direct capacity), semantic_2

Slopegraph = x in {Direct, CoT}, y = mean passphrase detection (report_contains),
one line per model, colored by family (validated categorical slots, direct
labels = secondary encoding). PNGs render on white so they read in both themes.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# family -> validated categorical slot (light hexes)
FAM_COLOR = {
    "Qwen2.5": "#2a78d6", "Mistral": "#eb6834", "Gemma2": "#1baf7a",
    "Gemma3": "#eda100", "Llama3": "#e87ba4", "Falcon3": "#008300",
    "Phi4": "#4a3aa7",
}
LOADS = ["trivial", "semantic_1", "semantic_2", "semantic_3", "semantic_4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def _slope(ax, rows, title, ylab=True):
    """rows: list of (label, family, direct, cot). Draw Direct->CoT slopegraph.

    Models at ceiling in BOTH regimes (direct>0.98 & cot>0.98) carry no signal
    -> dimmed grey and unlabelled, folded into one 'at ceiling' note so the
    interpretable models stay legible.
    """
    ax.set_title(title, fontsize=11, color=INK, pad=10, fontweight="bold")
    n_ceiling = 0
    for lab, fam, d, c in rows:
        ceiling = (d > 0.98 and c > 0.98)
        if ceiling:
            n_ceiling += 1
            ax.plot([0, 1], [d, c], "-", color="#bdbdba", lw=1.0, alpha=0.5,
                    marker="o", ms=4, mfc="#bdbdba", mec="white", mew=0.5, zorder=2)
            continue
        col = FAM_COLOR.get(fam, MUTED)
        ax.plot([0, 1], [d, c], "-", color=col, lw=1.8, alpha=0.95,
                marker="o", ms=6, mfc=col, mec="white", mew=0.8, zorder=3)
        ax.annotate(lab, (1, c), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=col, va="center", zorder=4)
    if n_ceiling:
        ax.annotate(f"{n_ceiling} model(s) at ceiling\n(Direct≈CoT≈1.0, no effect)",
                    (1, 1.0), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color="#8a8a86", va="center", style="italic", zorder=4)
    ax.set_xlim(-0.15, 1.62)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Direct", "CoT"], fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if ylab:
        ax.set_ylabel("passphrase detection", fontsize=9.5, color=MUTED)


def pooled_semantic(mt):
    sem = mt[(mt.k == 5) & (mt.load != "trivial") & (mt.load != "(none)")]
    return sem.groupby(["model", "family", "params_b"]).agg(
        direct=("direct", "mean"), cot=("cot", "mean")).reset_index()


def fig_paradox(mt):
    k1 = mt[mt.k == 1].sort_values("params_b")
    k5 = pooled_semantic(mt).sort_values("params_b")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    r1 = [(f"{r.model}", r.family, r.direct, r.cot) for _, r in k1.iterrows()]
    r5 = [(f"{r.model}", r.family, r.direct, r.cot) for _, r in k5.iterrows()]
    _slope(axes[0], r1, "k = 1  (single task)\nCoT is neutral / helpful")
    _slope(axes[1], r5, "k = 5  (4 load tasks + passphrase)\nCoT causes blindness", ylab=False)
    fig.suptitle("CoT-induced blindness — the paradox (unlimited CoT budget, passphrase last)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "Lines connect the same model. Down-slope Direct→CoT = CoT-induced blindness. "
             "Semantic loads pooled (k=5). Color = family.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    p = os.path.join(_HERE, "fig_cib_paradox.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_by_load(mt):
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.2), sharey=True)
    for ax, load in zip(axes, LOADS):
        sub = mt[(mt.k == 5) & (mt.load == load)].sort_values("params_b")
        rows = [(r.model, r.family, r.direct, r.cot) for _, r in sub.iterrows()]
        _slope(ax, rows, load, ylab=(load == "trivial"))
    fig.suptitle("CoT-induced blindness per load  (k=5, Direct→CoT, one line per model)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(_HERE, "fig_cib_by_load.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_vs_capacity(mt, load="semantic_2"):
    sub = mt[(mt.k == 5) & (mt.load == load)].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in sub.iterrows():
        col = FAM_COLOR.get(r.family, MUTED)
        ax.scatter(r.direct, r.cib, s=90, color=col, edgecolor="white",
                   linewidth=1, zorder=3)
        ax.annotate(r.model, (r.direct, r.cib), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, color=col)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_xlabel(f"Direct capacity (passphrase detection, {load})", color=INK)
    ax.set_ylabel("CoT-induced blindness  (Direct − CoT)", color=INK)
    ax.set_title(f"CoT-induced blindness vs Direct capacity — {load} (k=5)",
                 fontsize=12, fontweight="bold")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # family legend
    fams = sub["family"].unique()
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=FAM_COLOR.get(f, MUTED),
               mec="white", ms=8, label=f) for f in fams]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(_HERE, "fig_cib_vs_capacity.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    mt = pd.read_csv(os.path.join(_HERE, "cib_master_table.csv"))
    for fn in (fig_paradox, fig_by_load, fig_vs_capacity):
        print("wrote", fn(mt))


if __name__ == "__main__":
    main()
