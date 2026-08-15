"""One figure telling the gemma3:4b story:
 (a) report vs lag at 1024 (ceiling) and 256 (truncation collapse) in cot
 (b) graded log-prob vs lag (cot, direct) at 1024 — the truncation-immune measure.
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BOOL = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
COL = {"none": "#5b6472", "easy": "#2a9d3f", "hard": "#d1495b"}


def load(path):
    d = pd.read_csv(path)
    for c in ["report_correct", "t1_correct", "output_truncated",
              "t2_slot_missing", "thinking_is_placeholder"]:
        if c in d.columns and d[c].dtype == object:
            d[c] = d[c].map(BOOL)
    return d


def gated(d):
    g = (d.output_truncated.fillna(0) == 0) & (d.t2_slot_missing.fillna(0) == 0)
    g &= ~((d.regime == "cot") & (d.thinking_is_placeholder.fillna(0) == 1))
    return d[g]


d1024 = load(os.path.join(HERE, "ab_gemma3_4b_N50_t0_s50.csv"))   # powered, n=50
d256 = load(os.path.join(HERE, "ab_gemma3_4b_TRUNC256_t0_s10.csv"))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
loads = ["none", "easy", "hard"]


def line(a, df, reg, measure, gate=False, err=False):
    dd = gated(df) if gate else df
    dd = dd[dd.regime == reg]
    for ld in loads:
        grp = dd[dd.t1_load == ld].groupby("lag")[measure]
        s = grp.mean()
        e = grp.sem() if err else None
        a.errorbar(s.index, s.values, yerr=(e.values if err else None),
                   fmt="o-", color=COL[ld], capsize=3, label=f"T1={ld}")
    a.grid(alpha=.3)
    a.set_xlabel("lag")


line(ax[0, 0], d1024, "cot", "report_correct")
ax[0, 0].set_title("report vs lag — COT, budget=1024, n=50\n(current: at ceiling, no deficit)")
ax[0, 0].set_ylabel("P(T2 reported)"); ax[0, 0].set_ylim(-.05, 1.05); ax[0, 0].legend()

line(ax[0, 1], d256, "cot", "report_correct")
ax[0, 1].set_title("report vs lag — COT, budget=256\n(collapses for ALL loads = truncation artifact)")
ax[0, 1].set_ylabel("P(T2 reported)"); ax[0, 1].set_ylim(-.05, 1.05); ax[0, 1].legend()

line(ax[1, 0], d1024, "cot", "t2_mean_logprob", gate=True, err=True)
ax[1, 0].set_title("graded log-prob vs lag — COT, budget=1024, n=50 ±SEM\n(truncation-immune: no dip; load raises it → inverted)")
ax[1, 0].set_ylabel("mean log p(T2)"); ax[1, 0].legend()

line(ax[1, 1], d1024, "direct", "t2_mean_logprob", gate=True, err=True)
ax[1, 1].set_title("graded log-prob vs lag — DIRECT, budget=1024, n=50 ±SEM\n(flat — no structure)")
ax[1, 1].set_ylabel("mean log p(T2)"); ax[1, 1].legend()

fig.suptitle("gemma3:4b — the June 'blink' is a truncation artifact, not an attentional effect",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(HERE, "fig_4b_mechanism.png")
fig.savefig(out, dpi=115)
print("wrote", out)
