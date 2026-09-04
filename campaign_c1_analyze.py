"""Campaign C1 (issue #18 §6) — the FREE length analysis, on Campaign A output.

C1 asks, at fixed load: does a longer chain of thought go with a smaller chance
the passphrase is reported? It needs no new runs — Campaign A already logs, per
sample, both the report bit (`report_bits`) and the realised CoT length
(`cot_len_per`). This is the between-item correlation done *within* load × model,
i.e. Li et al. (2025) App. I redone properly on the per-sample level. It is still
correlational (length is endogenous — that is what C2's filler-packet instrument
is for); reported here as the naive estimate so the C1↔C2 gap is visible later.

Per (model, load) on the `cot` regime it fits a logistic regression
    report_bit ~ z(cot_len_chars)
pooling all K×n_trials samples, and reports the slope (log-odds per +1 SD of CoT
length) with a cluster-robust-ish trial bootstrap CI (resample trials, not
samples, so within-trial correlation doesn't shrink the interval). A negative
slope = longer CoT, less report = an interference/decay component.

Usage (from the folder CONTAINING LLM_Blink/):
    python3 LLM_Blink/campaign_c1_analyze.py LLM_Blink/results/scale_*.csv
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd

LOAD_ORDER = ["trivial", "math_bench_2", "math_bench_4", "math_bench_5"]


def _floats(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or cell == "":
        return []
    return [float(x) for x in str(cell).split(";") if x != ""]


def explode(df):
    """One row per (trial, sample) with report_bit and cot_len_chars."""
    rows = []
    for _, r in df.iterrows():
        if r["regime"] != "cot":
            continue
        rb = _floats(r.get("report_bits"))
        cl = _floats(r.get("cot_len_per"))
        if len(rb) != len(cl) or not rb:
            continue
        for j, (b, c) in enumerate(zip(rb, cl)):
            rows.append((r["t1_load"], int(r["seed"]), b, c))
    return pd.DataFrame(rows, columns=["load", "seed", "report", "cot_len"])


def logit_slope(x, y, iters=200):
    """Newton-fit intercept+slope of P(y=1) ~ logistic(a + b x). Returns b."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(np.unique(y)) < 2 or np.std(x) == 0:
        return np.nan
    a, b = 0.0, 0.0
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        W = np.clip(p * (1 - p), 1e-6, None)
        z = X.T @ (y - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, z)
        a += step[0]; b += step[1]
        if abs(step).max() < 1e-8:
            break
    return b


def trial_bootstrap(sub, n=1000, seed=0):
    """Bootstrap the logistic slope by resampling TRIALS (clusters)."""
    rng = np.random.default_rng(seed)
    trials = sub["seed"].unique()
    zmean, zstd = sub["cot_len"].mean(), sub["cot_len"].std() or 1.0
    def slope_of(d):
        return logit_slope((d["cot_len"] - zmean) / zstd, d["report"])
    point = slope_of(sub)
    bs = []
    by = {t: sub[sub.seed == t] for t in trials}
    for _ in range(n):
        pick = rng.choice(trials, len(trials), replace=True)
        d = pd.concat([by[t] for t in pick], ignore_index=True)
        s = slope_of(d)
        if s == s:
            bs.append(s)
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
    return point, lo, hi


def main():
    paths = sys.argv[1:] or sorted(glob.glob("LLM_Blink/results/scale_*.csv"))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    out = ["# Campaign C1 — realised CoT length vs passphrase report (naive, within load×model)\n",
           "slope = Δ log-odds(report) per +1 SD of CoT-length chars; negative = "
           "longer CoT, less report.\n"]
    for p in paths:
        df = pd.read_csv(p)
        model = os.path.basename(p).replace("scale_", "").replace(".csv", "").replace("_", ":", 1)
        ex = explode(df)
        if ex.empty:
            continue
        out.append(f"\n[{model}]")
        for load in [L for L in LOAD_ORDER if L in set(ex["load"])]:
            sub = ex[ex.load == load]
            if sub["report"].nunique() < 2:
                out.append(f"  {load:14s}  report constant ({sub['report'].mean():.2f}); slope n/a")
                continue
            b, lo, hi = trial_bootstrap(sub)
            sig = "" if (lo <= 0 <= hi) else "  *"
            out.append(f"  {load:14s}  n_trials={sub['seed'].nunique():3d}  "
                       f"report={sub['report'].mean():.2f}  "
                       f"slope={b:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
    text = "\n".join(out)
    print(text)
    with open("LLM_Blink/results/campaign_c1.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
