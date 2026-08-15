"""Diagnostic probe for one AB CSV: report / truncation / slot-missing / logprob
by lag, split by regime. Reused across the overnight gemma3:4b investigation.

Usage: python probe.py <csv> [title]
"""
import sys
import pandas as pd

BOOL = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
csv = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else csv

d = pd.read_csv(csv)
for c in ["report_correct", "t1_correct", "output_truncated",
          "t2_slot_missing", "thinking_is_placeholder"]:
    if c in d.columns and d[c].dtype == object:
        d[c] = d[c].map(BOOL)

g = (d.output_truncated.fillna(0) == 0) & (d.t2_slot_missing.fillna(0) == 0)
g &= ~((d.regime == "cot") & (d.thinking_is_placeholder.fillna(0) == 1))
gd = d[g]

print(f"### {title}")
print(f"rows {len(d)} | gated {len(gd)} ({len(gd)/len(d):.0%}) | "
      f"max_new_tokens={d.max_new_tokens.iloc[0] if 'max_new_tokens' in d else '?'}")

# UNGATED report (so truncation-driven failures are visible, not gated away)
for reg in ["cot", "direct"]:
    sub = d[d.regime == reg]
    if not len(sub):
        continue
    print(f"\n[{reg}] P(T2 reported) by lag — UNGATED (raw generation success):")
    print(sub.pivot_table(index="lag", columns="t1_load",
                          values="report_correct", aggfunc="mean").round(2).to_string())
    print(f"[{reg}] output_truncated rate by lag:")
    print(sub.pivot_table(index="lag", columns="t1_load",
                          values="output_truncated", aggfunc="mean").round(2).to_string())

# graded measure on gated data
for reg in ["cot", "direct"]:
    sub = gd[gd.regime == reg]
    if not len(sub):
        continue
    print(f"\n[{reg}] mean t2 log-prob by lag (gated):")
    print(sub.pivot_table(index="lag", columns="t1_load",
                          values="t2_mean_logprob", aggfunc="mean").round(2).to_string())

print("\nT1 acc by load (cot, gated):")
print(gd[gd.regime == "cot"].groupby("t1_load")["t1_correct"].mean().round(2).to_string())
