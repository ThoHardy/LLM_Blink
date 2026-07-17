"""Blink leaderboard: LoadCost and BlinkScore per model x regime (issue #9).

Definitions, computed on gated data (output_truncated == False,
t2_slot_missing == False, and for cot rows thinking_is_placeholder == False):

    Delta_regime(lag) = M(none, lag) - M(semantic_4, lag)

    LoadCost   = mean of Delta over all lags          (overall cost of the load)
    BlinkScore = mean Delta over lags (2, 4, 6)
                 - mean Delta over lags (0, 10)       (positive = lag-selective
                                                       deficit, i.e. blink-shaped)

M is P(report_correct) for the primary scores; the same two scores computed on
mean T2 log-prob are the secondary ``*_lp`` columns.

Usage
-----
    python LLM_Blink/leaderboard.py results/ab_*.csv \
        --out-table results/leaderboard.csv \
        --out-plot  results/blinkscore_vs_params.png

Model name and seeds are parsed from the file name (``ab_<slug>_t<T>_s<N>.csv``,
falling back to ``ab_results_<slug>.csv``); the parameter count is parsed from
the slug (``gemma2_9b`` -> 9.0 B, ``gemma3_270m`` -> 0.27 B).
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

MID_LAGS = (2, 4, 6)
EXT_LAGS = (0, 10)
LOAD = "semantic_4"
BASELINE = "none"


# -- quality gates -------------------------------------------------------------

def gate(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the issue #9 quality gates. Missing flag columns gate nothing."""
    keep = pd.Series(True, index=df.index)
    for col in ("output_truncated", "t2_slot_missing"):
        if col in df.columns:
            keep &= ~df[col].fillna(False).astype(bool)
    if "thinking_is_placeholder" in df.columns:
        is_cot = df["regime"] == "cot"
        keep &= ~(is_cot & df["thinking_is_placeholder"].fillna(False).astype(bool))
    return df[keep]


def retention(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell retention after gating, plus individual flag rates."""
    gated = gate(df)
    out = df.groupby(["regime", "t1_load"]).size().to_frame("n_raw")
    out["n_kept"] = gated.groupby(["regime", "t1_load"]).size()
    out["n_kept"] = out["n_kept"].fillna(0).astype(int)
    out["retention"] = out["n_kept"] / out["n_raw"]
    for col in ("output_truncated", "t2_slot_missing", "thinking_is_placeholder"):
        if col in df.columns:
            out[f"rate_{col}"] = (
                df.groupby(["regime", "t1_load"])[col]
                .apply(lambda s: s.fillna(False).astype(bool).mean())
            )
    return out.reset_index()


# -- scores --------------------------------------------------------------------

def delta_by_lag(df: pd.DataFrame, measure: str,
                 load: str = LOAD, baseline: str = BASELINE) -> pd.Series:
    """Delta(lag) = M(baseline, lag) - M(load, lag) for one regime's rows."""
    m = df.groupby(["t1_load", "lag"])[measure].mean().unstack("t1_load")
    for col in (baseline, load):
        if col not in m.columns:
            raise ValueError(f"t1_load={col!r} absent from data")
    return m[baseline] - m[load]


def load_cost(delta: pd.Series) -> float:
    return float(delta.mean())


def blink_score(delta: pd.Series,
                mid=MID_LAGS, ext=EXT_LAGS) -> float:
    have = set(delta.dropna().index)
    missing = (set(mid) | set(ext)) - have
    if missing:
        warnings.warn(f"BlinkScore computed without lags {sorted(missing)}")
    return float(delta.reindex(mid).mean() - delta.reindex(ext).mean())


def model_scores(df: pd.DataFrame, model: str, n_params_b: float | None) -> list[dict]:
    """One leaderboard row per regime, on gated data."""
    gated = gate(df)
    rows = []
    for regime, sub in gated.groupby("regime"):
        row: dict = {"model": model, "params_b": n_params_b, "regime": regime,
                     "n_trials_gated": len(sub)}
        d = delta_by_lag(sub, "report_correct")
        row["load_cost"] = load_cost(d)
        row["blink_score"] = blink_score(d)
        if "t2_mean_logprob" in sub.columns:
            d_lp = delta_by_lag(sub, "t2_mean_logprob")
            row["load_cost_lp"] = load_cost(d_lp)
            row["blink_score_lp"] = blink_score(d_lp)
        rows.append(row)
    return rows


# -- file-name parsing ---------------------------------------------------------

def parse_result_name(path: str | Path) -> str:
    """``ab_<slug>_t<T>_s<N>.csv`` or ``ab_results_<slug>.csv`` -> slug."""
    stem = Path(path).stem
    m = re.fullmatch(r"ab_(.+)_t[\d.]+_s\d+", stem)
    if m:
        return m.group(1)
    m = re.fullmatch(r"ab_results_(.+)", stem)
    if m:
        return m.group(1)
    return stem


def param_count_b(slug: str) -> float | None:
    """Parameter count in billions from a model slug; None if unparseable."""
    hits = re.findall(r"(\d+(?:\.\d+)?)([mb])(?=[_\W]|$)", slug.lower())
    if not hits:
        return None
    num, unit = hits[-1]
    return float(num) / (1000.0 if unit == "m" else 1.0)


# -- assembly ------------------------------------------------------------------

def build_leaderboard(files: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for f in files:
        slug = parse_result_name(f)
        rows.extend(model_scores(pd.read_csv(f), slug, param_count_b(slug)))
    lb = pd.DataFrame(rows)
    # rank by BlinkScore in cot (issue #9); models without cot rows sink last
    cot_rank = (lb[lb.regime == "cot"].set_index("model")["blink_score"]
                if "cot" in set(lb.regime) else pd.Series(dtype=float))
    lb["_rank_key"] = lb["model"].map(cot_rank)
    lb = (lb.sort_values(["_rank_key", "model", "regime"],
                         ascending=[False, True, True])
            .drop(columns="_rank_key").reset_index(drop=True))
    return lb


def plot_blinkscore_vs_params(lb: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for regime, sub in lb.dropna(subset=["params_b"]).groupby("regime"):
        sub = sub.sort_values("params_b")
        ax.plot(sub["params_b"], sub["blink_score"], marker="o", label=regime)
        for _, r in sub.iterrows():
            ax.annotate(r["model"], (r["params_b"], r["blink_score"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (B, log scale)")
    ax.set_ylabel("BlinkScore  [Delta(mid lags) - Delta(extreme lags)]")
    ax.set_title("BlinkScore vs model size, by regime")
    ax.legend(title="regime")
    ax.grid(alpha=0.3)
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Build the blink leaderboard from result CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csvs", nargs="+", help="Result CSVs (one per model).")
    parser.add_argument("--out-table", default=None,
                        help="Write the leaderboard table to this CSV.")
    parser.add_argument("--out-plot", default=None,
                        help="Write the BlinkScore-vs-params figure to this PNG.")
    parser.add_argument("--retention", action="store_true",
                        help="Also print per-file retention tables.")
    args = parser.parse_args()

    if args.retention:
        for f in args.csvs:
            print(f"\n== retention: {f} ==")
            print(retention(pd.read_csv(f)).to_string(index=False))

    lb = build_leaderboard(args.csvs)
    print("\n== leaderboard (ranked by BlinkScore in cot) ==")
    print(lb.to_string(index=False))

    if args.out_table:
        lb.to_csv(args.out_table, index=False)
        print(f"\nSaved table to {args.out_table}")
    if args.out_plot:
        import matplotlib
        matplotlib.use("Agg")
        ax = plot_blinkscore_vs_params(lb)
        ax.figure.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
