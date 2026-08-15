"""Cross-model replication figure (issue #14): the two-stage dissociation.

From the nested-probe CSVs of several models, one row of panels per model:
  left  : report_rate | passphrase IN vs NOT-IN the sampled CoT, per load, + ICC
          (all-or-none read-out gated by workspace entry).
  right : per-trial access-rate distribution (mean t2_in_cot over k_cot samples)
          = graded workspace entry.

Usage: python LLM_Blink/make_replication_figure.py out.png model1.csv model2.csv ...
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLM_Blink.mixture import icc_nested   # noqa: E402

C_IN, C_OUT, C_ACC = "#54A24B", "#E45756", "#F58518"


def _model_name(path):
    return os.path.basename(path).replace("nested_", "").replace(".csv", "")


def main(out, csvs):
    n = len(csvs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.2 * n), squeeze=False)
    for i, path in enumerate(csvs):
        nd = pd.read_csv(path)
        for c in ("s", "k", "cot_id", "seed", "t2_in_this_cot"):
            nd[c] = pd.to_numeric(nd[c], errors="coerce")
        name = _model_name(path)
        loads = sorted(nd.t1_load.unique())

        # left: ignition bars + ICC
        ax = axes[i][0]
        x = np.arange(len(loads)); w = 0.36
        iccs_for_title = []
        for j, L in enumerate(loads):
            g = nd[nd.t1_load == L]
            r_in = (g[g.t2_in_this_cot == 1].eval("s/k")).mean()
            r_out = (g[g.t2_in_this_cot == 0].eval("s/k")).mean()
            ids, y = [], []
            for _, r in g.iterrows():
                gid = int(r.seed) * 1000 + int(r.cot_id)
                ids += [gid] * int(r.k); y += [1] * int(r.s) + [0] * int(r.k - r.s)
            icc = icc_nested(ids, y); iccs_for_title.append(icc)
            ax.bar(j - w/2, r_in, w, color=C_IN,
                   label="passphrase IN CoT" if j == 0 else None)
            ax.bar(j + w/2, r_out, w, color=C_OUT,
                   label="passphrase NOT in CoT" if j == 0 else None)
            ax.text(j, 1.03, f"ICC={icc:.2f}", ha="center", fontsize=10, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(loads); ax.set_ylim(0, 1.16)
        ax.set_ylabel("report_rate | this CoT")
        mean_icc = float(np.mean(iccs_for_title))
        kind = ("read-out gated by CoT entry (ALL-OR-NONE)"
                if mean_icc >= 0.5 else "read-out a stochastic decision (GRADED)")
        ax.set_title(f"{name}: {kind}")
        ax.legend(loc="center left", fontsize=8)

        # right: per-trial access-rate distribution
        ax2 = axes[i][1]
        acc = nd.groupby(["t1_load", "seed"]).agg(
            a=("t2_in_this_cot", "mean")).reset_index()
        for L in loads:
            ax2.hist(acc[acc.t1_load == L].a, bins=np.linspace(0, 1, 11),
                     alpha=0.6, label=f"{L} (mean {acc[acc.t1_load==L].a.mean():.2f})")
        ax2.set_title(f"{name}: workspace ACCESS per trial (graded)")
        ax2.set_xlabel("fraction of sampled CoTs containing the passphrase")
        ax2.set_ylabel("trials"); ax2.legend(fontsize=8)

    fig.suptitle("Graded workspace access on every model; the READ-OUT's "
                 "discreteness is MODEL-DEPENDENT\n(gemma2:2b, qwen2.5:3b: "
                 "all-or-none read-out  ·  mistral:7b: graded read-out)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    out = sys.argv[1]
    main(out, sys.argv[2:])
