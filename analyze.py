"""Aggregate + plot the read-out curves.

Legacy figure: read-out vs lag, one line per T1 load (an AB-like signature =
a dip at intermediate lag for the hard load, absent for 'none').
Combined-design figures: read-out vs n_tasks (lines per finite_budget) and
vs finite_budget (lines per n_tasks) — pass ``x`` / ``by`` accordingly.
"""
from __future__ import annotations
import json

import pandas as pd
import matplotlib.pyplot as plt

from .readout import (answer_contains, normalize_answer,
                      parse_task_report_names, task_rows, cot_enumeration_stats)


def backfill_readout_columns(df: pd.DataFrame) -> pd.DataFrame:
    """RE-PARSE the free report of an old CSV with the current tolerant
    parser (no GPU needed) and recompute every parser-derived column.

    2026-07-21: the original parser missed answers not in ``- Task NAME: ...``
    form (e.g. quoted list items inside a ```python code fence, the Qwen0.5B
    direct-regime habit) — 17/20 direct+trivial trials of the second HAB pilot
    were logged as 0 tasks reported. This backfill re-runs
    ``parse_task_report_names`` on ``raw_output`` using the task names stored
    in the ``tasks`` JSON, then recomputes, for task-design rows:
    ``tasks`` (rebuilt rows incl. ``correct_lenient`` / ``answer_migrated``),
    ``report_correct``, ``report_contains``, ``n_tasks_reported``,
    ``n_hallucinated_tasks``, ``answer_migration``, ``t1_correct`` (fraction
    of load tasks answered exactly), and ``phrase_anywhere`` (all rows).
    2026-07-22: also recomputes the anti-enumeration compliance columns
    (``n_packets_in_cot``, ``n_filler_packets_in_cot``,
    ``enumerated_fillers_in_cot``) on task-design cot rows — retroactive on
    CSVs predating idea I1. Legacy rows (no tasks JSON) only get
    ``phrase_anywhere``.

    NOT recomputed (needs the model): ``t2_*`` log-probs and
    ``t2_slot_missing`` — the graded columns still reflect the slot found at
    run time. Returns a modified COPY.
    """
    df = df.copy()
    out_cols = {c: [] for c in (
        "tasks", "report_correct", "report_contains", "phrase_anywhere",
        "n_tasks_reported", "n_hallucinated_tasks", "answer_migration",
        "t1_correct", "n_packets_in_cot", "n_filler_packets_in_cot",
        "enumerated_fillers_in_cot")}
    for _, r in df.iterrows():
        raw = str(r.get("raw_output") or "")
        phrase = str(r.get("t2_phrase", "") or "")
        out_cols["phrase_anywhere"].append(
            bool(phrase) and phrase.upper() in raw.upper())
        tj = r.get("tasks")
        if not isinstance(tj, str) or not tj:            # legacy row
            for c in ("tasks", "report_correct", "report_contains",
                      "n_tasks_reported", "n_hallucinated_tasks",
                      "answer_migration", "t1_correct", "n_packets_in_cot",
                      "n_filler_packets_in_cot", "enumerated_fillers_in_cot"):
                out_cols[c].append(r.get(c))
            continue
        tasks = json.loads(tj)
        parsed = parse_task_report_names(raw, (t["name"] for t in tasks))
        rows = task_rows(tasks, parsed["reported"])
        pp = next(t for t in rows if t["kind"] == "passphrase")
        load_rows = [t for t in rows if t["kind"] == "load"]
        rep_rows = [t for t in rows if t["reported"]]
        out_cols["tasks"].append(json.dumps(rows))
        out_cols["report_correct"].append(bool(pp["correct"]))
        out_cols["report_contains"].append(bool(pp["correct_lenient"]))
        out_cols["n_tasks_reported"].append(len(parsed["reported"]))
        out_cols["n_hallucinated_tasks"].append(
            len(set(parsed["hallucinated"])))
        out_cols["answer_migration"].append(
            sum(t["answer_migrated"] for t in rep_rows) / len(rep_rows)
            if rep_rows else None)
        out_cols["t1_correct"].append(
            sum(t["correct"] for t in load_rows) / len(load_rows)
            if load_rows else r.get("t1_correct"))
        if str(r.get("regime")) == "cot":
            es = cot_enumeration_stats(raw, [t["packet"] for t in tasks])
        else:
            es = {"n_packets_in_cot": None, "n_filler_packets_in_cot": None,
                  "enumerated_fillers_in_cot": None}
        for c in ("n_packets_in_cot", "n_filler_packets_in_cot",
                  "enumerated_fillers_in_cot"):
            out_cols[c].append(es[c])
    for c, v in out_cols.items():
        df[c] = v
    return df


def summarize(df: pd.DataFrame, by=("t1_load", "lag")) -> pd.DataFrame:
    agg = {"t2_mean_logprob": ["mean", "sem"], "t2_joint_prob": ["mean", "sem"]}
    for c in ("report_correct", "report_contains", "phrase_anywhere"):
        if c in df.columns:
            df = df.assign(**{c: pd.to_numeric(df[c].map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}),
                errors="coerce")})
            agg[c] = ["mean", "sem"]
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
