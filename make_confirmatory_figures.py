"""Confirmatory-run analysis + figures (issue #14, Thomas's ask #3).

Reads the n=200 probe CSVs (probe_pilot output) for 2-3 models and produces:

  1. fig_confirm_reproduce.png  — THE priority figure: does CoT reproduce the
     passphrase-report collapse (the #10 dissociation)? Two panels, Direct->CoT,
     one line per model: (1) T1 load accuracy [CoT should help], (2) passphrase
     report rate [CoT should hurt].
  2. fig_confirm_per_model.png  — per model (Thomas's note: individual-model
     first): passphrase report_rate vs load (Direct vs CoT), and the cot
     report_rate count histogram stratified by realized report (all-or-none vs
     graded shape).
  3. confirm_stats.txt          — per-model numbers: dissociation, mixture CV
     verdict on the report counts across loads, Tarone Z, binary access/report
     level-2 count.

Usage (from the folder CONTAINING LLM_Blink/):
    python3 LLM_Blink/make_confirmatory_figures.py \
        LLM_Blink/results/confirm_gemma2_2b.csv \
        LLM_Blink/results/confirm_qwen2.5_3b.csv \
        LLM_Blink/results/confirm_mistral_7b.csv
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
from LLM_Blink.mixture import (betabinom_mixture_cv, verdict_from_cv,   # noqa: E402
                               overdispersion_z)

LOADS = ["trivial", "semantic_2", "semantic_4"]
LOADED = ["semantic_2", "semantic_4"]     # conditions with real load
MIN_N = 15                                 # skip a cell below this many trials
C_DIRECT, C_COT = "#4C78A8", "#E45756"


def _model(path):
    b = os.path.basename(path).replace("confirm_", "").replace(".csv", "")
    return b.replace("_", ":", 1) if b[0].isalpha() else b


def _load(path):
    df = pd.read_csv(path)
    for c in ("report_s", "report_k", "t1_correct", "realized_report_contains",
              "realized_t2_in_cot"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _rate(df):
    """Aggregate report rate = sum(s)/sum(k) over the rows (graded probe)."""
    k = df["report_k"].sum()
    return (df["report_s"].sum() / k) if k else np.nan


def reproduce_figure(dfs, names, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for df, name in zip(dfs, names):
        d = df[df["t1_load"].isin(LOADED)]
        pts = {}
        for reg in ("direct", "cot"):
            sub = d[d["regime"] == reg]
            if len(sub) < MIN_N:
                continue
            pts[reg] = (sub["t1_correct"].mean(), _rate(sub), len(sub))
        if {"direct", "cot"} <= set(pts):
            x = [0, 1]
            ax[0].plot(x, [pts["direct"][0], pts["cot"][0]], "-o", label=name)
            ax[1].plot(x, [pts["direct"][1], pts["cot"][1]], "-o", label=name)
    for a, title, ylab in ((ax[0], "T1 load accuracy\n(CoT should HELP)", "mean t1_correct"),
                           (ax[1], "Passphrase report\n(CoT should HURT = the blink)",
                            "report_rate  (s/k, T=1 fork)")):
        a.set_xticks([0, 1]); a.set_xticklabels(["Direct", "CoT"])
        a.set_title(title, fontsize=11); a.set_ylabel(ylab)
        a.set_ylim(-0.03, 1.03); a.grid(alpha=0.3); a.set_xlim(-0.2, 1.2)
    ax[1].legend(fontsize=9, title="model", loc="best")
    fig.suptitle("Reproducing the CoT-induced blindness on loaded conditions "
                 "(semantic_2+4, n=200/cell, T=1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def per_model_figure(dfs, names, out):
    n = len(dfs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.7 * n), squeeze=False)
    for r, (df, name) in enumerate(zip(dfs, names)):
        # -- col A: report_rate vs load, Direct vs CoT --
        axA = axes[r][0]
        for reg, c in (("direct", C_DIRECT), ("cot", C_COT)):
            ys, xs = [], []
            for i, load in enumerate(LOADS):
                sub = df[(df["t1_load"] == load) & (df["regime"] == reg)]
                if len(sub) >= MIN_N:
                    xs.append(i); ys.append(_rate(sub))
            axA.plot(xs, ys, "-o", color=c, label=reg)
        axA.set_xticks(range(len(LOADS))); axA.set_xticklabels(LOADS, fontsize=8)
        axA.set_ylim(-0.03, 1.03); axA.set_ylabel("report_rate"); axA.grid(alpha=0.3)
        axA.set_title(f"{name}: report_rate vs load", fontsize=10)
        axA.legend(fontsize=8)
        # -- col B: cot report_rate count histogram, loaded, stratified --
        axB = axes[r][1]
        cot = df[(df["regime"] == "cot") & (df["t1_load"].isin(LOADED))
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
            axB.set_title(f"{name}: cot report_rate shape (loaded)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def stats(dfs, names, out):
    lines = []
    def p(x): lines.append(x); print(x)
    for df, name in zip(dfs, names):
        p(f"\n===== {name} =====")
        for load in LOADS:
            row = []
            for reg in ("direct", "cot"):
                sub = df[(df["t1_load"] == load) & (df["regime"] == reg)]
                if len(sub) >= MIN_N:
                    row.append(f"{reg}: t1={sub['t1_correct'].mean():.2f} "
                               f"report_rate={_rate(sub):.2f} (n={len(sub)})")
            p(f"  [{load}] " + " | ".join(row))
        # binary access/report dissociation (level-2 cell): cot, loaded
        cot = df[(df["regime"] == "cot") & (df["t1_load"].isin(LOADED))]
        if len(cot):
            lvl2 = ((cot["realized_t2_in_cot"] == 1) &
                    (cot["realized_report_contains"] == 0)).mean()
            p(f"  access/report level-2 (in CoT, not reported): {lvl2:.2f} "
              f"of {len(cot)} loaded cot trials")
        # mixture CV on report counts across loads (D2: pi vs mu with load)
        cotp = cot[cot["report_k"].notna() & (cot["report_k"] > 0)]
        conds = cotp["t1_load"].tolist()
        if len(cotp) >= 60 and len(set(conds)) >= 2:
            try:
                cv = betabinom_mixture_cv(cotp["report_s"].values,
                                          cotp["report_k"].values, conds,
                                          strat=cotp["realized_report_contains"].values,
                                          folds=5, seeds=3, boot=1000)
                v = verdict_from_cv(cv)
                z = overdispersion_z(cotp["report_s"].values,
                                     cotp["report_k"].values, conds)
                p(f"  report-stage mixture verdict: {v['call']}")
                p(f"  Tarone Z (overdispersion): {z:.1f}")
            except Exception as e:
                p(f"  mixture: skipped ({e})")
        else:
            p(f"  mixture: need >=60 loaded cot trials over >=2 loads "
              f"(have {len(cotp)})")
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
    R = os.path.join(_HERE, "results")
    reproduce_figure(dfs, names, os.path.join(R, "fig_confirm_reproduce.png"))
    per_model_figure(dfs, names, os.path.join(R, "fig_confirm_per_model.png"))
    stats(dfs, names, os.path.join(R, "confirm_stats.txt"))


if __name__ == "__main__":
    main(sys.argv[1:])
