"""Analyze the CoT-induced blindness (CIB) sweep — master table + controls.

Loads every results/cib_<model>.csv, re-backfills the tolerant parser, and
builds the per-(model, load) table at k=5:
  direct_report (parallel-readout ceiling), cot_report, CIB = direct - cot,
  direct_t1 / cot_t1 (control 1: does CoT help the load task?), n, trunc.
Plus the k=1 contrast (single passphrase task) per model.

Gating (issue #10): report/detection analyses gate on output_truncated==False
ONLY (t2_slot_missing is the detection OUTCOME here, gating on it is circular).
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from LLM_Blink import backfill_readout_columns  # noqa: E402

LOADS = ["trivial", "semantic_1", "semantic_2", "semantic_3", "semantic_4"]


def load_all(pattern=None):
    pattern = pattern or os.path.join(_HERE, "cib_*.csv")
    frames = []
    for f in sorted(glob.glob(pattern)):
        df = pd.read_csv(f, low_memory=False)
        if "model" not in df.columns:
            df["model"] = os.path.basename(f)[4:-4]
        frames.append(backfill_readout_columns(df))
    if not frames:
        raise SystemExit("no cib_*.csv found")
    return pd.concat(frames, ignore_index=True)


def master_table(df):
    g = df[df["output_truncated"] == False]
    recs = []
    for (model, pb, fam), md in g.groupby(["model", "params_b", "family"]):
        k5 = md[md["n_tasks"] == 5]
        for load in LOADS:
            c = k5[(k5["t1_load"] == load) & (k5["regime"] == "cot")]
            d = k5[(k5["t1_load"] == load) & (k5["regime"] == "direct")]
            if not len(c) or not len(d):
                continue
            recs.append(dict(
                model=model, params_b=pb, family=fam, load=load, k=5,
                direct=d["report_contains"].mean(),
                cot=c["report_contains"].mean(),
                cib=d["report_contains"].mean() - c["report_contains"].mean(),
                direct_t1=d["t1_correct"].mean(),
                cot_t1=c["t1_correct"].mean(),
                t1_gain_cot=c["t1_correct"].mean() - d["t1_correct"].mean(),
                n=min(len(c), len(d)),
            ))
        # k=1 contrast
        k1 = md[md["n_tasks"] == 1]
        c1, d1 = k1[k1["regime"] == "cot"], k1[k1["regime"] == "direct"]
        if len(c1) and len(d1):
            recs.append(dict(
                model=model, params_b=pb, family=fam, load="(none)", k=1,
                direct=d1["report_contains"].mean(),
                cot=c1["report_contains"].mean(),
                cib=d1["report_contains"].mean() - c1["report_contains"].mean(),
                direct_t1=np.nan, cot_t1=np.nan, t1_gain_cot=np.nan,
                n=min(len(c1), len(d1)),
            ))
    return pd.DataFrame(recs)


def main():
    df = load_all()
    print(f"loaded {len(df)} rows, {df['model'].nunique()} models, "
          f"trunc rate {df['output_truncated'].mean():.3f}\n")
    mt = master_table(df)
    mt.to_csv(os.path.join(_HERE, "cib_master_table.csv"), index=False)

    # ---- per-load leaderboards (k=5), sorted by CIB desc ------------------
    for load in LOADS:
        sub = mt[(mt["k"] == 5) & (mt["load"] == load)].sort_values("cib", ascending=False)
        print(f"\n### semantic-load = {load}  (k=5, CoT-induced blindness = direct - cot)")
        print(f"{'model':18}{'p':>5} | {'direct':>7}{'cot':>7}{'CIB':>7} | "
              f"{'dir_t1':>7}{'cot_t1':>7}{'t1gain':>7} | {'interp?':>8}")
        for _, r in sub.iterrows():
            # interpretable if the load task is doable (direct t1 mid/high) AND
            # detection has headroom in direct (< .97) so CIB can be non-zero
            interp = (r["direct_t1"] >= 0.3) and (r["direct"] <= 0.98)
            print(f"{r['model']:18}{r['params_b']:>5.0f} | "
                  f"{r['direct']:>7.2f}{r['cot']:>7.2f}{r['cib']:>7.2f} | "
                  f"{r['direct_t1']:>7.2f}{r['cot_t1']:>7.2f}{r['t1_gain_cot']:>+7.2f} | "
                  f"{'yes' if interp else 'CEIL/weak':>8}")

    # ---- control 1: does CoT help the load task (t1)? ---------------------
    print("\n\n### CONTROL 1 — CoT effect on load-task accuracy t1 (k=5, semantic loads pooled)")
    sem = mt[(mt["k"] == 5) & (mt["load"] != "trivial")]
    ctl = sem.groupby(["model", "params_b"]).agg(
        direct_t1=("direct_t1", "mean"), cot_t1=("cot_t1", "mean"),
        t1_gain_cot=("t1_gain_cot", "mean")).reset_index().sort_values("params_b")
    print(f"{'model':18}{'p':>5} | {'dir_t1':>7}{'cot_t1':>7}{'gain':>7}  (gain>0 = CoT helps -> effect well-posed)")
    for _, r in ctl.iterrows():
        flag = "" if r["t1_gain_cot"] > 0 else "  <-- CoT does NOT help load; effect ill-posed here"
        print(f"{r['model']:18}{r['params_b']:>5.0f} | {r['direct_t1']:>7.2f}{r['cot_t1']:>7.2f}{r['t1_gain_cot']:>+7.2f}{flag}")

    # ---- k=1 vs k=5 CIB (the paradox) ------------------------------------
    print("\n\n### k=1 vs k=5 CoT-induced blindness (semantic loads pooled for k=5)")
    k1 = mt[mt["k"] == 1].set_index("model")["cib"]
    k5 = sem.groupby("model")["cib"].mean()
    both = pd.DataFrame({"k1_cib": k1, "k5_cib": k5}).dropna().sort_values("k5_cib", ascending=False)
    print(both.round(3).to_string())


if __name__ == "__main__":
    main()
