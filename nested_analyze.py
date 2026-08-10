"""Analyse a nested-probe CSV (issue #14 §4.3): ICC of report within a CoT.

ICC ~ 1  -> report deterministic given a CoT (ignition / all-or-none access).
ICC << 1 -> read-out itself stochastic (graded).

Also reports, per load: the between-CoT and within-CoT variance of report_rate,
and the fraction of sampled CoTs that are "decided" (report_rate <=0.1 or >=0.9)
vs "undecided" (in between) — a legibility check on where the stochasticity sits.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from LLM_Blink.mixture import icc_nested                            # noqa: E402


def analyse(path):
    df = pd.read_csv(path)
    for c in ("s", "k", "cot_id", "seed", "t2_in_this_cot"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print("=" * 70 + f"\nNESTED PROBE: {path}\n" + "=" * 70)
    print(f"rows(CoTs): {len(df)}  trials: {df.groupby(['t1_load','seed']).ngroups}"
          f"  loads: {sorted(df.t1_load.unique())}")
    for L, g in df.groupby("t1_load"):
        # global CoT id and per-report binary reconstruction
        cot_ids, y = [], []
        for _, r in g.iterrows():
            gid = int(r.seed) * 1000 + int(r.cot_id)
            cot_ids += [gid] * int(r.k)
            y += [1] * int(r.s) + [0] * int(r.k - r.s)
        icc = icc_nested(cot_ids, y)
        rr = (g.s / g.k)
        # between = var of per-CoT means; within = mean of per-CoT binomial var
        between = rr.var(ddof=1)
        within = (rr * (1 - rr)).mean()
        decided = ((rr <= 0.1) | (rr >= 0.9)).mean()
        acc = g.t2_in_this_cot.mean()
        print(f"\n  {L}:  ICC={icc:.3f}   (1=ignition, 0=graded read-out)")
        print(f"    between-CoT var={between:.3f}  within-CoT var={within:.3f}  "
              f"n_CoTs={len(g)}")
        print(f"    fraction of CoTs 'decided' (rr<=.1 or >=.9): {decided:.2f}"
              f"   mean access(t2_in_CoT)={acc:.2f}")
        # report_rate conditioned on whether the passphrase was in that CoT
        for v, sub in g.groupby("t2_in_this_cot"):
            lbl = "passphrase IN CoT" if v == 1 else "passphrase NOT in CoT"
            print(f"    report_rate | {lbl:22}: {(sub.s/sub.k).mean():.3f} (n={len(sub)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv")
    analyse(ap.parse_args().csv)
