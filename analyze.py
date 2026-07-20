"""Aggregate + plot the read-out curves.

Legacy figure: read-out vs lag, one line per T1 load (an AB-like signature =
a dip at intermediate lag for the hard load, absent for 'none').
Combined-design figures: read-out vs n_tasks (lines per finite_budget) and
vs finite_budget (lines per n_tasks) — pass ``x`` / ``by`` accordingly.
"""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def summarize(df: pd.DataFrame, by=("t1_load", "lag")) -> pd.DataFrame:
    agg = {"t2_mean_logprob": ["mean", "sem"], "t2_joint_prob": ["mean", "sem"]}
    if "report_correct" in df.columns:
        agg["report_correct"] = ["mean", "sem"]
    g = df.groupby(list(by), dropna=False).agg(agg)
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index()


_XLABELS = {
    "lag": "lag (packets between T1 and T2)",
    "n_tasks": "n_tasks (tasks per stream, incl. the passphrase task)",
    "finite_budget": "finite CoT budget (tokens inside <Thinking>)",
}


def plot_ab(df: pd.DataFrame, measure: str = "t2_mean_logprob",
            regime: str | None = None, ax=None,
            x: str = "lag", by: str = "t1_load"):
    """measure: 't2_mean_logprob' (graded/unconscious) or 'report_correct'
    (binary/conscious). ``x`` is the sweep axis, ``by`` the line grouping;
    defaults reproduce the legacy lag figure. Pass regime='cot' or 'direct'
    to plot one regime; default pools all rows in df.
    """
    if regime is not None and "regime" in df.columns:
        df = df[df["regime"] == regime]
    s = summarize(df, by=(by, x))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for key, sub in s.groupby(by, dropna=False):
        sub = sub.sort_values(x)
        y, e = f"{measure}_mean", f"{measure}_sem"
        ax.errorbar(sub[x], sub[y], yerr=sub[e], marker="o", capsize=3,
                    label=f"{by}={key}")
    ax.set_xlabel(_XLABELS.get(x, x))
    ylabel = {"t2_mean_logprob": "mean log p(T2)  [graded 'unconscious']",
              "report_correct": "P(T2 reported)  [binary 'conscious']"}.get(measure, measure)
    ax.set_ylabel(ylabel)
    rtxt = f"  (regime={regime})" if regime else ""
    ax.set_title(f"LLM Attentional Blink: T2 read-out vs {x}{rtxt}")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
