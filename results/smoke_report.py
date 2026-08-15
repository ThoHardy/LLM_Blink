"""Smoke-test report for a single AB results CSV.

Usage: python smoke_report.py <csv> [runtime_seconds]
Prints quality-flag rates, t1_correct per load, and report/logprob summaries.
"""
import sys
import pandas as pd

csv = sys.argv[1]
runtime = float(sys.argv[2]) if len(sys.argv) > 2 else None
df = pd.read_csv(csv)
n = len(df)

# Bool columns may carry NaN (e.g. t1_correct for load='none'); coerce to 0/1 float.
_boolmap = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
for c in ["report_correct", "t1_correct", "output_truncated",
          "t2_slot_missing", "thinking_is_placeholder"]:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].map(_boolmap)

print(f"### {csv}")
print(f"rows: {n}")
if runtime is not None:
    print(f"runtime: {runtime:.0f}s total, {runtime / n:.2f}s/trial")

print("\n-- quality flags (mean rate) --")
flags = ["output_truncated", "t2_slot_missing", "thinking_is_placeholder"]
print(df.groupby("regime")[flags].mean().round(3).to_string())

print("\n-- t1_correct by load --")
print(df.groupby("t1_load")["t1_correct"].mean().round(3).to_string())

# Gated data (issue's gate)
gate = (~df.output_truncated.astype(bool)) & (~df.t2_slot_missing.astype(bool))
gate_cot = gate & ~((df.regime == "cot") & df.thinking_is_placeholder.astype(bool))
dfok = df[gate_cot]
print(f"\n-- retention after gate: {len(dfok)}/{n} = {len(dfok)/n:.2%} --")

print("\n-- report_correct by load x regime (gated) --")
if len(dfok):
    print(dfok.groupby(["t1_load", "regime"])["report_correct"].mean().round(3).to_string())

print("\n-- t2_mean_logprob by load x regime (gated) --")
if len(dfok):
    print(dfok.groupby(["t1_load", "regime"])["t2_mean_logprob"].mean().round(3).to_string())
print()
