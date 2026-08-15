"""Figures + blink leaderboard for whatever model CSVs exist so far.

For each results/ab_<slug>_t0_s10.csv:
  - gate rows (output_truncated==0, t2_slot_missing==0, cot: thinking_is_placeholder==0)
  - save a 2x2 gated figure: rows {report_correct, t2_mean_logprob} x cols {cot, direct}
Leaderboard (gated), per model x regime:
  Delta(lag) = P(report|none,lag) - P(report|semantic_4,lag)
  LoadCost   = mean Delta over all lags
  BlinkScore = mean Delta over {2,4,6} - mean Delta over {0,10}
  (+ same two on t2_mean_logprob as secondary)
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from LLM_Blink.analyze import plot_ab  # noqa: E402

BOOL = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
MID, EXT = [2, 4, 6], [0, 10]


def load_gated(path):
    df = pd.read_csv(path)
    for c in ["report_correct", "t1_correct", "output_truncated",
              "t2_slot_missing", "thinking_is_placeholder"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(BOOL)
    gate = (df.output_truncated.fillna(0) == 0) & (df.t2_slot_missing.fillna(0) == 0)
    gate &= ~((df.regime == "cot") & (df.thinking_is_placeholder.fillna(0) == 1))
    return df, df[gate].copy()


def blink_scores(dfok, measure):
    """Return dict regime -> (LoadCost, BlinkScore) for a measure where higher
    'none' minus 'semantic_4' = load deficit."""
    out = {}
    for regime in ["cot", "direct"]:
        d = dfok[dfok.regime == regime]
        piv = d.groupby(["t1_load", "lag"])[measure].mean()
        res = {}
        for lag in sorted(d.lag.unique()):
            try:
                res[lag] = piv[("none", lag)] - piv[("semantic_4", lag)]
            except KeyError:
                res[lag] = np.nan
        deltas = pd.Series(res)
        loadcost = deltas.mean()
        mid = deltas.reindex(MID).mean()
        ext = deltas.reindex(EXT).mean()
        out[regime] = (loadcost, mid - ext)
    return out


def model_label(slug):
    return slug.replace("ab_", "").replace("_t0_s10", "").replace("_", ":", 1)


def main():
    csvs = sorted(glob.glob(os.path.join(HERE, "ab_*_t0_s10.csv")))
    rows = []
    for path in csvs:
        slug = os.path.basename(path).replace(".csv", "")
        name = model_label(slug)
        df, dfok = load_gated(path)
        ret = len(dfok) / len(df) if len(df) else 0
        print(f"\n=== {name}  (gated {len(dfok)}/{len(df)} = {ret:.0%}) ===")

        # figure (gated)
        if len(dfok):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
            for col, regime in enumerate(["cot", "direct"]):
                plot_ab(dfok, "report_correct", regime=regime, ax=axes[0][col])
                plot_ab(dfok, "t2_mean_logprob", regime=regime, ax=axes[1][col])
            fig.suptitle(f"{name}  (gated n={len(dfok)})", fontsize=13)
            fig.tight_layout()
            out_png = os.path.join(HERE, f"fig_{slug}.png")
            fig.savefig(out_png, dpi=110)
            plt.close(fig)
            print(f"  figure -> {os.path.relpath(out_png)}")

        # leaderboard
        rep = blink_scores(dfok, "report_correct")
        lp = blink_scores(dfok, "t2_mean_logprob")
        for regime in ["cot", "direct"]:
            rows.append(dict(
                model=name, regime=regime,
                LoadCost_report=rep[regime][0], BlinkScore_report=rep[regime][1],
                LoadCost_logp=lp[regime][0], BlinkScore_logp=lp[regime][1],
                gated_n=len(dfok),
                report_none=dfok[(dfok.regime == regime) & (dfok.t1_load == "none")]["report_correct"].mean(),
                report_sem4=dfok[(dfok.regime == regime) & (dfok.t1_load == "semantic_4")]["report_correct"].mean(),
            ))

    board = pd.DataFrame(rows)
    board.to_csv(os.path.join(HERE, "leaderboard_so_far.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n\n================ BLINK LEADERBOARD (so far, gated) ================")
    print(board.round(3).to_string(index=False))
    print("\nBlinkScore>0 = lag-selective deficit (blink-shaped); ~0 = flat/none.")


if __name__ == "__main__":
    main()
