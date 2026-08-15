"""Two-stage dissociation figure (issue #14): graded access vs all-or-none report.

Panels:
  A  report_rate distribution (cot vs direct) — the forked report probe. cot piles
     at 0 and k (all-or-none), direct piles high.
  B  access_rate distribution (cot) — the forked access probe. Mass in the middle
     = graded workspace entry.
  C  nested probe: report_rate GIVEN a sampled CoT, split by whether the passphrase
     was in that CoT, per load — the ignition bars + ICC.
  D  access taxonomy (levels 1/2/3) per load.

Colourblind-safe palette, readable on white. Usage:
  python LLM_Blink/make_two_stage_figure.py <probe_csv> <nested_csv> [out.png]
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_COT, C_DIR, C_ACC = "#4C78A8", "#B0B0B0", "#F58518"
C_IN, C_OUT = "#54A24B", "#E45756"


def _num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(probe_csv, nested_csv, out):
    df = _num(pd.read_csv(probe_csv),
              ["report_s", "report_k", "access_s", "access_k",
               "realized_report_contains"])
    nd = _num(pd.read_csv(nested_csv), ["s", "k", "cot_id", "seed", "t2_in_this_cot"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # --- A: report_rate distribution, cot vs direct
    cot = df[(df.regime == "cot") & (df.report_k > 0)]
    dirr = df[(df.regime == "direct") & (df.report_k > 0)]
    K = int(df.report_k.dropna().median())
    bins = np.arange(-0.5, K + 1.5, 1)
    ax[0, 0].hist(cot.report_s, bins=bins, alpha=0.75, color=C_COT,
                  label=f"cot (n={len(cot)})", density=True)
    ax[0, 0].hist(dirr.report_s, bins=bins, alpha=0.6, color=C_DIR,
                  label=f"direct (n={len(dirr)})", density=True)
    ax[0, 0].set_title("A. Report probe: p(report) per trial\n"
                       "cot piles at 0 & k (all-or-none); direct piles high")
    ax[0, 0].set_xlabel(f"successes out of k={K}"); ax[0, 0].set_ylabel("density")
    ax[0, 0].legend()

    # --- B: access_rate distribution, cot
    acc = df[(df.regime == "cot") & (df.access_k > 0)]
    if len(acc):
        ax[0, 1].hist(acc.access_s, bins=bins, color=C_ACC,
                      label=f"cot access (n={len(acc)})", density=True)
    ax[0, 1].set_title("B. Access probe: p(enters CoT) per trial\n"
                       "mass in the middle = GRADED workspace entry")
    ax[0, 1].set_xlabel(f"successes out of k={K}"); ax[0, 1].set_ylabel("density")
    ax[0, 1].legend()

    # --- C: nested ignition bars + ICC
    from LLM_Blink.mixture import icc_nested
    loads = sorted(nd.t1_load.unique())
    x = np.arange(len(loads)); w = 0.36
    r_in, r_out, iccs = [], [], []
    for L in loads:
        g = nd[nd.t1_load == L]
        r_in.append((g[g.t2_in_this_cot == 1].eval("s/k")).mean())
        r_out.append((g[g.t2_in_this_cot == 0].eval("s/k")).mean())
        ids, y = [], []
        for _, r in g.iterrows():
            gid = int(r.seed) * 1000 + int(r.cot_id)
            ids += [gid] * int(r.k); y += [1] * int(r.s) + [0] * int(r.k - r.s)
        iccs.append(icc_nested(ids, y))
    ax[1, 0].bar(x - w/2, r_in, w, color=C_IN, label="passphrase IN sampled CoT")
    ax[1, 0].bar(x + w/2, r_out, w, color=C_OUT, label="passphrase NOT in CoT")
    for i, ic in enumerate(iccs):
        ax[1, 0].text(i, 1.02, f"ICC={ic:.2f}", ha="center", fontsize=11, weight="bold")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(loads)
    ax[1, 0].set_ylim(0, 1.15); ax[1, 0].set_ylabel("report_rate | this CoT")
    ax[1, 0].set_title("C. Nested probe: report is decided by the CoT\n"
                       "(ICC high = all-or-none read-out gated by workspace entry)")
    ax[1, 0].legend(loc="center left")

    # --- D: access taxonomy
    if len(acc):
        a = acc.copy(); a["rr"] = a.report_s / a.report_k; a["ar"] = a.access_s / a.access_k
        lv = {L: [] for L in ["1", "2", "3"]}
        loads2 = sorted(a.t1_load.unique())
        for L in loads2:
            g = a[a.t1_load == L]
            l1 = ((g.rr >= 0.5)).mean()
            l2 = ((g.rr < 0.5) & (g.ar >= 0.5)).mean()
            l3 = ((g.rr < 0.5) & (g.ar < 0.5)).mean()
            lv["1"].append(l1); lv["2"].append(l2); lv["3"].append(l3)
        xb = np.arange(len(loads2))
        ax[1, 1].bar(xb, lv["1"], color=C_IN, label="1: accessed & reported")
        ax[1, 1].bar(xb, lv["2"], bottom=lv["1"], color=C_ACC,
                     label="2: accessed, NOT reported")
        ax[1, 1].bar(xb, lv["3"], bottom=np.add(lv["1"], lv["2"]), color=C_OUT,
                     label="3: not accessed")
        ax[1, 1].set_xticks(xb); ax[1, 1].set_xticklabels(loads2)
        ax[1, 1].set_ylabel("fraction of trials")
        ax[1, 1].set_title("D. Access taxonomy (level 2 = the GWT-key cell)")
        ax[1, 1].legend(fontsize=8)

    fig.suptitle("CoT-induced blindness: graded workspace ACCESS, all-or-none READ-OUT "
                 f"(gemma2:2b)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    probe = sys.argv[1] if len(sys.argv) > 1 else "results/probe_gemma2_2b.csv"
    nested = sys.argv[2] if len(sys.argv) > 2 else "results/nested_gemma2_2b.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "results/fig_two_stage_gemma2_2b.png"
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    main(probe, nested, out)
