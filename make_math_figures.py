"""MATH-load run figures (issue #14, Thomas's MATH ask).

Same three deliverables as `make_confirmatory_figures.py` but (a) load-agnostic
(loads are read from the CSVs, so it works on the Hendrycks MATH-Algebra levels
`math_bench_*`) and (b) with a THIRD figure Thomas asked for explicitly: the
fitted (mu, pi) mixture parameters and how the MATH load level modulates them.

Outputs (into LLM_Blink/results/):
  1. fig_math_reproduce.png    — CoT should HELP T1 accuracy (the point of using
     MATH loads) yet HURT passphrase report. One line per model, Direct->CoT.
  2. fig_math_per_model.png    — per model: report_rate vs load (Direct vs CoT),
     and the cot report-count histogram stratified by realized report
     (all-or-none vs graded shape).
  3. fig_math_mixture_mu_pi.png — per model, the Mfull beta-binomial fit's
     component means mu_lo/mu_hi (left axis) and high-mode weight pi (right axis)
     vs MATH load. pi-shift w/ flat mu = ignition/all-or-none; mu-slide = graded.
  4. math_stats.txt

Usage (from the folder CONTAINING LLM_Blink/):
    python3 LLM_Blink/make_math_figures.py LLM_Blink/results/math_*.csv
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from LLM_Blink.mixture import (fit_betabinom_mixture, overdispersion_z)  # noqa: E402

MIN_N = 15
C_DIRECT, C_COT = "#4C78A8", "#E45756"
C_LO, C_HI, C_PI = "#54A24B", "#E45756", "#4C78A8"


def _model(path):
    b = os.path.basename(path).replace("math_", "").replace("select_", "")
    b = b.replace(".csv", "")
    return b.replace("_", ":", 1) if b and b[0].isalpha() else b


def _load(path):
    df = pd.read_csv(path)
    for c in ("report_s", "report_k", "t1_correct", "realized_report_contains",
              "realized_t2_in_cot"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _loads_in(dfs):
    """Ordered union of loads across CSVs: trivial first, then math_bench_1..5,
    then anything else alphabetically."""
    present = set()
    for df in dfs:
        present |= set(df["t1_load"].unique())
    order = (["trivial"] + [f"math_bench_{i}" for i in range(1, 6)]
             + [f"semantic_{i}" for i in range(0, 5)])
    ordered = [l for l in order if l in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _rate(df):
    k = df["report_k"].sum()
    return (df["report_s"].sum() / k) if k else np.nan


def reproduce_figure(dfs, names, loads, out):
    loaded = [l for l in loads if l != "trivial"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for df, name in zip(dfs, names):
        d = df[df["t1_load"].isin(loaded)]
        pts = {}
        for reg in ("direct", "cot"):
            sub = d[d["regime"] == reg]
            if len(sub) < MIN_N:
                continue
            pts[reg] = (sub["t1_correct"].mean(), _rate(sub))
        if {"direct", "cot"} <= set(pts):
            ax[0].plot([0, 1], [pts["direct"][0], pts["cot"][0]], "-o", label=name)
            ax[1].plot([0, 1], [pts["direct"][1], pts["cot"][1]], "-o", label=name)
    for a, title, ylab in (
            (ax[0], "T1 MATH-load accuracy\n(CoT should HELP)", "mean t1_correct"),
            (ax[1], "Passphrase report\n(CoT should HURT = the blink)",
             "report_rate  (s/k, T=1 fork)")):
        a.set_xticks([0, 1]); a.set_xticklabels(["Direct", "CoT"])
        a.set_title(title, fontsize=11); a.set_ylabel(ylab)
        a.set_ylim(-0.03, 1.03); a.grid(alpha=0.3); a.set_xlim(-0.2, 1.2)
    ax[1].legend(fontsize=9, title="model", loc="best")
    fig.suptitle("CoT-induced blindness on MATH-Algebra loads "
                 "(Hendrycks levels, T=1): CoT helps the load, blinds the passphrase",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def per_model_figure(dfs, names, loads, out):
    loaded = [l for l in loads if l != "trivial"]
    n = len(dfs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.7 * n), squeeze=False)
    for r, (df, name) in enumerate(zip(dfs, names)):
        axA = axes[r][0]
        for reg, c in (("direct", C_DIRECT), ("cot", C_COT)):
            xs, ys = [], []
            for i, load in enumerate(loads):
                sub = df[(df["t1_load"] == load) & (df["regime"] == reg)]
                if len(sub) >= MIN_N:
                    xs.append(i); ys.append(_rate(sub))
            axA.plot(xs, ys, "-o", color=c, label=reg)
        axA.set_xticks(range(len(loads)))
        axA.set_xticklabels(loads, fontsize=8, rotation=20)
        axA.set_ylim(-0.03, 1.03); axA.set_ylabel("report_rate"); axA.grid(alpha=0.3)
        axA.set_title(f"{name}: report_rate vs load", fontsize=10); axA.legend(fontsize=8)
        axB = axes[r][1]
        cot = df[(df["regime"] == "cot") & (df["t1_load"].isin(loaded))
                 & df["report_k"].notna()]
        if len(cot):
            kk = int(cot["report_k"].median())
            for lab, mask, c in (
                    ("realized: reported", cot["realized_report_contains"] == 1, "#59A14F"),
                    ("realized: not reported", cot["realized_report_contains"] == 0, "#B07AA1")):
                s = cot.loc[mask, "report_s"].dropna()
                if len(s):
                    axB.hist(s, bins=np.arange(-0.5, kk + 1.5), alpha=0.55,
                             color=c, label=f"{lab} (n={len(s)})")
            axB.set_xlabel(f"report successes s (of k={kk})")
            axB.set_ylabel("trials"); axB.legend(fontsize=8)
            axB.set_title(f"{name}: cot report shape (loaded)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def mixture_figure(dfs, names, loads, out):
    """Item 5: fitted (mu, pi) and how MATH load modulates them, per model."""
    loaded = [l for l in loads if l != "trivial"]
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for j, (df, name) in enumerate(zip(dfs, names)):
        ax = axes[0][j]
        cot = df[(df["regime"] == "cot") & (df["t1_load"].isin(loaded))
                 & df["report_k"].notna() & (df["report_k"] > 0)].copy()
        # keep only loads with enough trials, in load order
        used = [l for l in loaded
                if (cot["t1_load"] == l).sum() >= MIN_N]
        cot = cot[cot["t1_load"].isin(used)]
        if len(used) < 2 or len(cot) < 40:
            ax.set_title(f"{name}: too few trials"); ax.axis("off"); continue
        s = cot["report_s"].values
        k = cot["report_k"].values
        cond = cot["t1_load"].values
        fit = fit_betabinom_mixture(s, k, cond, model="Mfull", n_restarts=8, seed=0)
        order = [fit.conditions.index(l) for l in used]
        x = np.arange(len(used))
        mu_lo = fit.mu[0][order]; mu_hi = fit.mu[1][order]; pi = fit.pi[order]
        ax.plot(x, mu_lo, "-o", color=C_LO, label=r"$\mu_{low}$ (blink mode)")
        ax.plot(x, mu_hi, "-o", color=C_HI, label=r"$\mu_{high}$ (report mode)")
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel(r"component mean $\mu$")
        ax.set_xticks(x); ax.set_xticklabels(used, fontsize=8, rotation=20)
        ax.grid(alpha=0.3); ax.set_title(f"{name}", fontsize=11)
        ax2 = ax.twinx()
        ax2.plot(x, pi, "--s", color=C_PI, label=r"$\pi$ (high-mode weight)")
        ax2.set_ylim(-0.03, 1.03); ax2.set_ylabel(r"$\pi_{high}$", color=C_PI)
        ax2.tick_params(axis="y", labelcolor=C_PI)
        if j == 0:
            l1, la1 = ax.get_legend_handles_labels()
            l2, la2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, la1 + la2, fontsize=8, loc="center left")
    fig.suptitle("Beta-binomial mixture (Mfull): does MATH load move pi (ignition) "
                 "or mu (resource-sharing)?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def stats(dfs, names, loads, out):
    loaded = [l for l in loads if l != "trivial"]
    lines = []
    def p(x): lines.append(x); print(x)
    for df, name in zip(dfs, names):
        p(f"\n===== {name} =====")
        for load in loads:
            row = []
            for reg in ("direct", "cot"):
                sub = df[(df["t1_load"] == load) & (df["regime"] == reg)]
                if len(sub) >= MIN_N:
                    row.append(f"{reg}: t1={sub['t1_correct'].mean():.2f} "
                               f"report_rate={_rate(sub):.2f} (n={len(sub)})")
            if row:
                p(f"  [{load}] " + " | ".join(row))
        cot = df[(df["regime"] == "cot") & (df["t1_load"].isin(loaded))
                 & df["report_k"].notna() & (df["report_k"] > 0)].copy()
        if len(cot):
            k = cot["report_k"].values
            s = cot["report_s"].values
            extreme = np.mean((s <= 1) | (s >= k - 1))
            p(f"  report SHAPE: {extreme:.0%} at extremes -> "
              f"{'U-shaped / all-or-none' if extreme > 0.6 else 'graded / mixed'}")
            rr = s / k
            inc = cot["realized_t2_in_cot"] == 1
            if inc.any() and (~inc).any():
                p(f"  read-out gap: |in-CoT={rr[inc].mean():.2f} "
                  f"vs |not-in-CoT={rr[~inc].mean():.2f} "
                  f"({rr[inc].mean() - rr[~inc].mean():+.2f})")
            conds = cot["t1_load"].tolist()
            if len(cot) >= 60 and len(set(conds)) >= 2:
                z = overdispersion_z(s, k, conds)
                p(f"  Tarone Z: {z['Z']:.1f}")
                fit = fit_betabinom_mixture(s, k, cot["t1_load"].values,
                                            model="Mfull", n_restarts=8, seed=0)
                for i, c in enumerate(fit.conditions):
                    p(f"    {c}: pi_hi={fit.pi[i]:.2f} "
                      f"mu_lo={fit.mu[0][i]:.2f} mu_hi={fit.mu[1][i]:.2f}")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", out)


def main(csvs):
    dfs, names = [], []
    for c in csvs:
        if os.path.exists(c):
            dfs.append(_load(c)); names.append(_model(c))
    if not dfs:
        print("no CSVs found"); return
    loads = _loads_in(dfs)
    print("loads:", loads)
    R = os.path.join(_HERE, "results")
    reproduce_figure(dfs, names, loads, os.path.join(R, "fig_math_reproduce.png"))
    per_model_figure(dfs, names, loads, os.path.join(R, "fig_math_per_model.png"))
    mixture_figure(dfs, names, loads, os.path.join(R, "fig_math_mixture_mu_pi.png"))
    stats(dfs, names, loads, os.path.join(R, "math_stats.txt"))


if __name__ == "__main__":
    main(sys.argv[1:])
