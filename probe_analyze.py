"""Analyse a forked-resampling probe CSV (issue #14): validity, shape, verdict.

Reads a ``probe_pilot.py`` output CSV and produces:

  1. validity checks (5): fork coverage, degenerate rate, calibration
     (mean report_rate ~ realized binary report at T=1), self-consistency
     (report_rate high on trials where the passphrase WAS realized);
  2. the raw count histograms per load, stratified on realized report — the
     first figure, before any fitting (4: U-shaped=ignition, middle=graded);
  3. the beta-binomial mixture CV model comparison + verdict on the REPORT
     counts (the primary probe) and on the ACCESS counts, across the load axis;
  4. the 4.2.3 pi->mu null band, Tarone's Z, and the 5 access taxonomy counts.

No GPU. Robust to partial data (min-n per cell configurable). Prints a report
and, unless --no-fig, writes histograms to results/.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from LLM_Blink.mixture import (betabinom_mixture_cv, verdict_from_cv,       # noqa: E402
                               fit_betabinom_mixture, pi_shift_null,
                               overdispersion_z, access_level)


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _load(path):
    df = pd.read_csv(path)
    for c in ("report_s", "report_k", "access_s", "access_k",
              "realized_report_contains", "realized_t2_in_cot",
              "has_fork_point", "report_degenerate", "access_degenerate"):
        if c in df:
            df[c] = _num(df, c)
    return df


def validity(df):
    print("\n" + "=" * 70 + "\nVALIDITY CHECKS (5)\n" + "=" * 70)
    print(f"rows: {len(df)}   loads: {sorted(df.t1_load.unique())}   "
          f"regimes: {sorted(df.regime.unique())}")
    print(f"per cell n:\n{df.groupby(['t1_load','regime']).size().to_string()}")
    fk = df.has_fork_point.mean()
    print(f"\n[1] report fork coverage: {fk:.3f}  (want ~1.0)")
    have = df[df.report_k > 0]
    deg = have.report_degenerate.mean() if len(have) else float('nan')
    print(f"[2] report degenerate rate: {deg:.3f}  (want ~0 -> seeds effective)")
    have = have.assign(rr=have.report_s / have.report_k)
    print("[3] calibration  mean(report_rate) vs mean(realized binary), by cell:")
    cal = have.groupby(["t1_load", "regime"]).apply(
        lambda g: pd.Series(dict(report_rate=g.rr.mean(),
                                 realized=g.realized_report_contains.mean(),
                                 n=len(g))), include_groups=False)
    print(cal.round(3).to_string())
    cot = have[have.regime == "cot"]
    if len(cot):
        hi = cot[cot.realized_report_contains == 1].rr.mean()
        lo = cot[cot.realized_report_contains == 0].rr.mean()
        print(f"[4] self-consistency (cot): report_rate | realized=1 -> {hi:.3f} "
              f"(want high) ; | realized=0 -> {lo:.3f}")
    print("[5] every stratified analysis below conditions on realized report.")


def _counts(df, regime, s_col, k_col, min_n=20):
    d = df[(df.regime == regime) & (df[k_col] > 0)].copy()
    d = d.dropna(subset=[s_col, k_col])
    keep = d.groupby("t1_load").filter(lambda g: len(g) >= min_n)
    return keep


def mixture_report(df, s_col="report_s", k_col="report_k", regime="cot",
                   label="REPORT", folds=8, seeds=3, boot=1500):
    print("\n" + "=" * 70 + f"\nMIXTURE: {label} probe, regime={regime}, "
          f"condition=load\n" + "=" * 70)
    d = _counts(df, regime, s_col, k_col)
    if d.t1_load.nunique() < 2:
        print("  not enough loads with data yet."); return None
    s = d[s_col].to_numpy(); k = d[k_col].to_numpy(); cond = d["t1_load"].tolist()
    strat = d["realized_report_contains"].fillna(0).astype(int).to_numpy()
    print(f"  n={len(s)}  per load: {d.groupby('t1_load').size().to_dict()}")
    print("  raw count means (s/k) by load:",
          {L: round((g[s_col]/g[k_col]).mean(), 3)
           for L, g in d.groupby('t1_load')})
    cv = betabinom_mixture_cv(s, k, cond, strat=strat, folds=folds,
                              seeds=seeds, boot=boot)
    print("  held-out logdens:", {m: round(x, 4) for m, x in cv["mean"].items()})
    for key, dd in cv["pairs"].items():
        star = "*" if dd["excludes_zero"] else " "
        print(f"    {key:>12}: {dd['mean']:+.4f} CI[{dd['ci'][0]:+.4f},{dd['ci'][1]:+.4f}] {star}")
    z = overdispersion_z(s, k, cond)
    print(f"  Tarone Z (overdispersion): {z['Z']:.1f}")
    v = verdict_from_cv(cv)
    print(f"  >>> VERDICT: {v['call']}")
    ff = cv["fits"]["Mfull"]
    print(f"  Mfull mu(lo)={np.round(ff.mu[0],3).tolist()} mu(hi)={np.round(ff.mu[1],3).tolist()}")
    print(f"        pi={np.round(ff.pi,3).tolist()}  (loads={ff.conditions})")
    # pi->mu null band on the extreme load pair
    npc = [int((d.t1_load == L).sum()) for L in ff.conditions]
    try:
        band = pi_shift_null(ff, int(np.median(k)), npc, cond_pair=(0, -1), n_rep=300)
        dmu = float(ff.mu[1][-1] - ff.mu[1][0])
        inside = band["band"][0] <= dmu <= band["band"][1]
        print(f"  pi->mu null band Δμ_hi(extreme loads): {np.round(band['band'],3)} "
              f"| observed {dmu:+.3f} -> {'inside (no graded claim)' if inside else 'OUTSIDE (graded signal)'}")
    except Exception as e:
        print("  pi->mu band skipped:", e)
    return cv


def access_taxonomy(df, hi=0.5, lo=0.5):
    print("\n" + "=" * 70 + "\nACCESS TAXONOMY (5): report x access, cot only\n" + "=" * 70)
    d = df[(df.regime == "cot") & (df.report_k > 0) & (df.access_k > 0)].copy()
    if not len(d):
        print("  no cot rows with both probes yet."); return
    d["rr"] = d.report_s / d.report_k
    d["ar"] = d.access_s / d.access_k
    d["level"] = access_level(d.rr, d.ar, hi=hi, lo=lo)
    lab = {1: "1 accessed&reported", 2: "2 accessed,NOT reported", 3: "3 not accessed"}
    for L, g in d.groupby("t1_load"):
        vc = g.level.value_counts(normalize=True)
        print(f"  {L:12}: " + "  ".join(
            f"{lab[i].split()[0]}={vc.get(i,0):.2f}" for i in (1, 2, 3)))
    print("  (level 2 = the theoretically-loaded 'accessed but not reported' cell)")
    print("  mean access_rate vs report_rate by load (cot):")
    print(d.groupby("t1_load")[["ar", "rr"]].mean().round(3).to_string())


def figures(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df.regime == "cot") & (df.report_k > 0)].copy()
    if not len(d):
        return
    loads = sorted(d.t1_load.unique())
    fig, axes = plt.subplots(2, len(loads), figsize=(4 * len(loads), 7), squeeze=False)
    K = int(d.report_k.median())
    for j, L in enumerate(loads):
        g = d[d.t1_load == L]
        axes[0][j].hist(g.report_s, bins=np.arange(-0.5, K + 1.5, 1), color="#4C78A8")
        axes[0][j].set_title(f"REPORT counts | {L}\n(cot, n={len(g)})")
        axes[0][j].set_xlabel(f"successes out of k={K}")
        if "access_k" in g and g.access_k.notna().any():
            ga = g[g.access_k > 0]
            axes[1][j].hist(ga.access_s, bins=np.arange(-0.5, K + 1.5, 1), color="#F58518")
            axes[1][j].set_title(f"ACCESS counts | {L} (n={len(ga)})")
            axes[1][j].set_xlabel(f"successes out of k={K}")
    fig.suptitle("Forked-resampling count histograms (U-shape=ignition, middle=graded)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_probe_histograms.png")
    fig.savefig(p, dpi=110); print(f"\n[fig] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    df = _load(a.csv)
    validity(df)
    mixture_report(df, "report_s", "report_k", "cot", "REPORT")
    mixture_report(df, "report_s", "report_k", "direct", "REPORT(direct baseline)")
    if "access_k" in df and df.access_k.notna().any():
        mixture_report(df, "access_s", "access_k", "cot", "ACCESS")
    access_taxonomy(df)
    if not a.no_fig:
        figures(df, os.path.dirname(os.path.abspath(a.csv)))


if __name__ == "__main__":
    main()
