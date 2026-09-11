"""§8.3 robustness check — is the entry distribution a rank-2 mixture? (issue #18)

Bimodality of a graded score is not invariant to monotone rescaling (Overgaard
2006; Nieuwenhuis & de Kleijn 2011). The invariant version of the all-or-none
claim: under it the per-trial entry distribution at load c is

    F_c = pi_c * F_hi + (1 - pi_c) * F_lo,     F_hi, F_lo fixed across c,

so only the mixing weight pi_c moves with load and the (bins x loads) matrix of
binned distributions is **rank 2**. Under a graded account the mode itself slides
with load and the rank exceeds 2.

Implementation (per model, on the cot-regime s_access counts, needs >=4 loads):
  1. bin s_access/k into B bins, one empirical distribution per load -> B x L matrix;
  2. centre it (subtract the row mean across loads) and take the singular spectrum;
  3. the all-or-none signature is that sigma_3 / sigma_2 is small (the centred
     matrix is ~rank 1 after removing the mean, i.e. rank 2 with the mean) — a
     single mode-weight direction explains the load-to-load variation;
  4. bootstrap over trials for a CI on sigma_3/sigma_2, and compare to a graded
     null (sliding mode) simulated at the observed per-load means.

s_report is analysed the same way as a secondary read-out. Read-only on the CSVs.
Usage:  python3 LLM_Blink/campaign_rank2.py LLM_Blink/results/scale_*.csv
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd

LOAD_ORDER = ["trivial", "math_bench_2", "math_bench_4", "math_bench_5"]
NBINS = 6


def dist_matrix(df, col, loads, k=20, nbins=NBINS):
    """B x L matrix: column l = binned empirical distribution of col/k at load l."""
    edges = np.linspace(0, 1, nbins + 1)
    M = np.zeros((nbins, len(loads)))
    for l, load in enumerate(loads):
        v = df[(df.t1_load == load) & (df.regime == "cot")][col].dropna().values
        if len(v) == 0:
            M[:, l] = np.nan; continue
        h, _ = np.histogram(np.asarray(v, float) / k, bins=edges)
        M[:, l] = h / max(1, h.sum())
    return M


def sigma_ratio(M):
    """sigma_3/sigma_2 of the column-centred matrix (small => rank-2 story holds)."""
    if np.isnan(M).any():
        return np.nan
    C = M - M.mean(axis=1, keepdims=True)
    s = np.linalg.svd(C, compute_uv=False)
    if len(s) < 3 or s[1] < 1e-12:
        return 0.0
    return float(s[2] / s[1])


def boot_ratio(df, col, loads, k=20, n=400, seed=0):
    rng = np.random.default_rng(seed)
    point = sigma_ratio(dist_matrix(df, col, loads, k))
    seeds = df["seed"].unique()
    by = {s: df[df.seed == s] for s in seeds}
    rs = []
    for _ in range(n):
        pick = rng.choice(seeds, len(seeds), replace=True)
        d = pd.concat([by[s] for s in pick], ignore_index=True)
        r = sigma_ratio(dist_matrix(d, col, loads, k))
        if r == r:
            rs.append(r)
    lo, hi = (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (np.nan, np.nan)
    return point, lo, hi


def graded_null_ratio(df, col, loads, k=20, n=200, seed=1):
    """Simulate a GRADED process matched to the observed per-load means: each trial's
    count ~ Binomial(k, p_load) with p_load = observed mean rate. A sliding-mean
    graded generator; its sigma_3/sigma_2 is the reference the all-or-none data
    should fall BELOW."""
    rng = np.random.default_rng(seed)
    means, ns = [], []
    for load in loads:
        v = df[(df.t1_load == load) & (df.regime == "cot")][col].dropna().values
        means.append(np.mean(v) / k if len(v) else np.nan)
        ns.append(len(v))
    if any(m != m for m in means):
        return np.nan, np.nan
    rs = []
    for _ in range(n):
        cols = []
        edges = np.linspace(0, 1, NBINS + 1)
        M = np.zeros((NBINS, len(loads)))
        for l, (p, nn) in enumerate(zip(means, ns)):
            draws = rng.binomial(k, p, size=max(nn, 1)) / k
            h, _ = np.histogram(draws, bins=edges); M[:, l] = h / max(1, h.sum())
        r = sigma_ratio(M)
        if r == r:
            rs.append(r)
    return (float(np.mean(rs)), float(np.std(rs))) if rs else (np.nan, np.nan)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("LLM_Blink/results/scale_*.csv"))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    out = ["# §8.3 rank-2 robustness check on the entry distribution\n",
           "sigma3/sigma2 of the column-centred (bins x loads) matrix. Small (and "
           "below the graded null) => the load-to-load change is a single mixing-weight "
           "direction = all-or-none, rank-2. Large => sliding mode = graded.\n"]
    for p in paths:
        df = pd.read_csv(p)
        for c in ("s_access", "s_report", "k", "seed"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        model = os.path.basename(p).replace("scale_", "").replace(".csv", "").replace("_", ":", 1)
        loads = [L for L in LOAD_ORDER if L in set(df["t1_load"])]
        if len(loads) < 4:
            out.append(f"\n[{model}]  <4 loads present ({loads}) — skip (§8.3 needs >=4)")
            continue
        out.append(f"\n[{model}]  loads={loads}")
        k = int(df["k"].median())
        for col in ("s_access", "s_report"):
            if col not in df or df[(df.regime=='cot')][col].dropna().empty:
                out.append(f"  {col:9s}  (no cot data)"); continue
            pt, lo, hi = boot_ratio(df, col, loads, k)
            gm, gs = graded_null_ratio(df, col, loads, k)
            verdict = ""
            if pt == pt and gm == gm:
                verdict = "rank-2/all-or-none" if pt < gm else "graded-like"
            out.append(f"  {col:9s}  sigma3/sigma2={pt:.3f} [{lo:.3f},{hi:.3f}]"
                       f"   graded-null={gm:.3f}±{gs:.3f}   -> {verdict}")
    text = "\n".join(out)
    print(text)
    with open("LLM_Blink/results/campaign_rank2.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
