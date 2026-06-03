"""Aggregate + plot the AB curve.

The diagnostic figure: read-out vs lag, one line per T1 load. An AB-like signature = a DIP at
intermediate lag for the 'hard' load that is absent (flat) for 'none'. Always compare to the
'none' load, which is the positional/recency baseline.
"""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def summarize(df: pd.DataFrame, by=("t1_load", "lag")) -> pd.DataFrame:
    agg = {"t2_mean_logprob": ["mean", "sem"], "t2_joint_prob": ["mean", "sem"]}
    if "report_correct" in df.columns:
        agg["report_correct"] = ["mean", "sem"]
    g = df.groupby(list(by)).agg(agg)
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index()


def plot_ab(df: pd.DataFrame, measure: str = "t2_mean_logprob", regime: str | None = None, ax=None):
    """measure: 't2_mean_logprob' (graded/unconscious) or 'report_correct' (binary/conscious).

    Pass regime='cot' or 'direct' to plot one regime; default pools all rows in df.
    """
    if regime is not None and "regime" in df.columns:
        df = df[df["regime"] == regime]
    s = summarize(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for load, sub in s.groupby("t1_load"):
        sub = sub.sort_values("lag")
        y, e = f"{measure}_mean", f"{measure}_sem"
        ax.errorbar(sub["lag"], sub[y], yerr=sub[e], marker="o", capsize=3, label=f"T1={load}")
    ax.set_xlabel("lag (packets between T1 and T2)")
    ylabel = {"t2_mean_logprob": "mean log p(T2)  [graded 'unconscious']",
              "report_correct": "P(T2 reported)  [binary 'conscious']"}.get(measure, measure)
    ax.set_ylabel(ylabel)
    rtxt = f"  (regime={regime})" if regime else ""
    ax.set_title(f"LLM Attentional Blink: T2 read-out vs lag{rtxt}")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
