"""Campaign A figures (issue #18 §4, figs 2/3/6/7) from the resample-full CSVs.

Consumes ``results/scale_*.csv`` (the FIELDS_FULL schema: per-trial s_access,
s_report, k, per-sample vectors). Renders whatever models are on disk, so it is
run repeatedly while the ladder fills in.

Outputs (results/):
  scale_stats.txt        — §2.1 truncation per cell FIRST (a >2% cell is failed),
                           then per model x load report/access/blink with CIs.
  fig3_blink_vs_size.png — blink = report(direct) - report(cot) vs params (log x),
                           one line per family, per load; floor models annotated.
  fig2_slopegraph.png    — load accuracy UP / passphrase report DOWN, direct->cot.
  fig6_border_hists.png  — s_access and s_report histograms along the ladder,
                           both borders from the same trials; overdispersion Z.
  fig7_leaderboards.png  — sorted blink (left) and fraction-at-extremes (right).

Usage (from the folder CONTAINING LLM_Blink/):
    python3 LLM_Blink/make_scale_figures.py LLM_Blink/results/scale_*.csv
"""
from __future__ import annotations
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from LLM_Blink.mixture import overdispersion_z                     # noqa: E402

LOAD_ORDER = ["trivial", "math_bench_2", "math_bench_4", "math_bench_5"]
FAM_COLORS = {"qwen2.5": "#4C78A8", "gemma2": "#E45756", "gemma3": "#54A24B",
              "llama3.1": "#B279A2", "mistral": "#EECA3B"}
FLOOR_REPORT = 0.80        # §4: direct report rate < 0.8 -> floor-compressed


def parse_model(path):
    b = os.path.basename(path).replace("scale_", "").replace(".csv", "")
    m = re.match(r"([a-z0-9.]+?)_([0-9.]+)b$", b)
    if m:
        fam, size = m.group(1), float(m.group(2))
        return f"{fam}:{size:g}b", fam, size
    return b, b.split("_")[0], np.nan


def load_all(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        name, fam, size = parse_model(p)
        df["model_name"] = name
        df["family"] = fam
        df["params_b"] = size
        for c in ("s_report", "s_access", "k", "t1_mean", "frac_truncated",
                  "n_truncated_samples"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def boot_ci(vals, stat=np.mean, n=2000, seed=0):
    vals = np.asarray([v for v in vals if v == v], dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return (float(stat(vals)), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def report_rate(sub):
    """mean over trials of s_report/k (per-trial rate, so trials weight equally)."""
    r = (sub["s_report"] / sub["k"]).dropna()
    return r.values


def access_rate(sub):
    r = (sub["s_access"] / sub["k"]).dropna()
    return r.values


def cells_table(df):
    """Per model x load: direct/cot report rate, access rate, blink, with CIs."""
    rows = []
    for name in sorted(df["model_name"].unique(), key=lambda n: df[df.model_name==n]["params_b"].iloc[0]):
        md = df[df.model_name == name]
        fam = md["family"].iloc[0]; size = md["params_b"].iloc[0]
        for load in [L for L in LOAD_ORDER if L in set(md["t1_load"])]:
            d = md[(md.t1_load == load) & (md.regime == "direct")]
            c = md[(md.t1_load == load) & (md.regime == "cot")]
            rd = report_rate(d); rc = report_rate(c); ac = access_rate(c)
            dm, dlo, dhi = boot_ci(rd)
            cm, clo, chi = boot_ci(rc)
            am, alo, ahi = boot_ci(ac)
            blink = dm - cm if (dm == dm and cm == cm) else np.nan
            rows.append(dict(model=name, family=fam, params_b=size, load=load,
                             n_direct=len(rd), n_cot=len(rc),
                             report_direct=dm, report_cot=cm, blink=blink,
                             access_cot=am,
                             report_direct_lo=dlo, report_direct_hi=dhi,
                             report_cot_lo=clo, report_cot_hi=chi,
                             floor=int(dm < FLOOR_REPORT if dm == dm else 0)))
    return pd.DataFrame(rows)


def truncation_report(df):
    lines = ["## §2.1 truncation gate (a cell > 2% is FAILED — re-run bigger)\n"]
    worst = 0.0
    for name in sorted(df["model_name"].unique()):
        md = df[df.model_name == name]
        lines.append(f"[{name}]  quant={md['quant_tag'].iloc[0] if 'quant_tag' in md else '?'}")
        for load in [L for L in LOAD_ORDER if L in set(md["t1_load"])]:
            for reg in ("cot", "direct"):
                sub = md[(md.t1_load == load) & (md.regime == reg)]
                if not len(sub):
                    continue
                frac = 100 * sub["n_truncated_samples"].sum() / (sub["k"].sum())
                worst = max(worst, frac)
                flag = "  <-- FAILED" if frac > 2 else ""
                lines.append(f"    {load:16s} {reg:7s} trunc={frac:5.2f}%"
                             f" (trials={len(sub)}){flag}")
    lines.append(f"\nworst cell truncation across all models: {worst:.2f}%\n")
    return "\n".join(lines)


def fig3(tab, out):
    loads = [L for L in LOAD_ORDER if L in set(tab["load"])]
    fig, axes = plt.subplots(1, len(loads), figsize=(4 * len(loads), 4.2),
                             sharey=True, squeeze=False)
    for j, load in enumerate(loads):
        ax = axes[0][j]
        t = tab[tab.load == load]
        for fam in sorted(t["family"].unique()):
            ft = t[t.family == fam].sort_values("params_b")
            ax.plot(ft["params_b"], ft["blink"], "-o",
                    color=FAM_COLORS.get(fam, "#888"), label=fam)
            for _, r in ft.iterrows():
                if r["floor"]:
                    ax.annotate("floor", (r["params_b"], r["blink"]),
                                fontsize=7, color="#c00",
                                xytext=(0, 6), textcoords="offset points", ha="center")
        ax.set_xscale("log")
        ax.axhline(0, color="#bbb", lw=.8)
        ax.set_title(load); ax.set_xlabel("params (B, log)")
        if j == 0:
            ax.set_ylabel("blink = report(direct) − report(cot)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Fig 3 — blink vs model size, within family "
                 "(prediction: interior peak; floor models are artefacts)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig2(df, tab, out):
    """load accuracy UP / passphrase report DOWN, direct->cot, per model."""
    names = sorted(df["model_name"].unique(),
                   key=lambda n: df[df.model_name == n]["params_b"].iloc[0])
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4.5))
    x = np.arange(len(names))
    # report drop (cot vs direct) and t1 rise (cot vs direct) at the hardest shared load
    rep_d, rep_c, t1_d, t1_c = [], [], [], []
    for name in names:
        md = df[df.model_name == name]
        load = [L for L in reversed(LOAD_ORDER) if L in set(md["t1_load"])][0]
        d = md[(md.t1_load == load) & (md.regime == "direct")]
        c = md[(md.t1_load == load) & (md.regime == "cot")]
        rep_d.append(np.nanmean(report_rate(d)) if len(d) else np.nan)
        rep_c.append(np.nanmean(report_rate(c)) if len(c) else np.nan)
        t1_d.append(np.nanmean(d["t1_mean"]) if len(d) else np.nan)
        t1_c.append(np.nanmean(c["t1_mean"]) if len(c) else np.nan)
    ax.plot(x, rep_d, "o--", color="#E45756", alpha=.5, label="report direct")
    ax.plot(x, rep_c, "o-", color="#E45756", label="report cot")
    ax.plot(x, t1_d, "s--", color="#54A24B", alpha=.5, label="load-acc direct")
    ax.plot(x, t1_c, "s-", color="#54A24B", label="load-acc cot")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rate"); ax.set_ylim(-.02, 1.02); ax.legend(fontsize=8)
    ax.set_title("Fig 2 — CoT lifts load accuracy while it lowers passphrase report "
                 "(hardest shared load)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig6(df, out):
    names = sorted(df["model_name"].unique(),
                   key=lambda n: df[df.model_name == n]["params_b"].iloc[0])
    loads = [L for L in LOAD_ORDER if L in set(df["t1_load"])]
    fig, axes = plt.subplots(len(names), len(loads),
                             figsize=(3 * len(loads), 2.4 * len(names)),
                             squeeze=False)
    for i, name in enumerate(names):
        for j, load in enumerate(loads):
            ax = axes[i][j]
            c = df[(df.model_name == name) & (df.t1_load == load) & (df.regime == "cot")]
            k = int(c["k"].median()) if len(c) else 20
            if len(c):
                sr = c["s_report"].dropna().values
                sa = c["s_access"].dropna().values
                bins = np.arange(-0.5, k + 1.5)
                ax.hist(sr, bins=bins, color="#E45756", alpha=.6, label="report")
                ax.hist(sa, bins=bins, color="#4C78A8", alpha=.5, label="access")
            if i == 0:
                ax.set_title(load, fontsize=9)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.set_xticks([0, k // 2, k])
            ax.tick_params(labelsize=7)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Fig 6 — s_access (blue) & s_report (red) per trial, same trajectories "
                 "(bimodal = all-or-none entry)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig7(tab, df, out):
    load = [L for L in reversed(LOAD_ORDER) if L in set(tab["load"])][0]
    t = tab[tab.load == load].sort_values("blink", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.5 + 0.5 * max(4, len(t))))
    y = np.arange(len(t))
    a1.barh(y, t["blink"], color=[FAM_COLORS.get(f, "#888") for f in t["family"]])
    a1.set_yticks(y); a1.set_yticklabels(t["model"], fontsize=8); a1.invert_yaxis()
    a1.set_xlabel("blink"); a1.set_title(f"Fig 7 left — blink-resistant board ({load})")
    a1.axvline(0, color="#bbb", lw=.8)
    # right board: fraction of cot trials with s_access at the extremes (all-or-none)
    ext = []
    for _, r in t.iterrows():
        c = df[(df.model_name == r["model"]) & (df.t1_load == load) & (df.regime == "cot")]
        sa = (c["s_access"] / c["k"]).dropna().values
        frac = np.mean((sa <= 0.1) | (sa >= 0.9)) if len(sa) else np.nan
        ext.append(frac)
    a2.barh(y, ext, color=[FAM_COLORS.get(f, "#888") for f in t["family"]])
    a2.set_yticks(y); a2.set_yticklabels(t["model"], fontsize=8); a2.invert_yaxis()
    a2.set_xlabel("fraction of trials with s_access at 0/1 extreme")
    a2.set_title("Fig 7 right — all-or-none entry")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("LLM_Blink/results/scale_*.csv"))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    if not paths:
        print("no scale_*.csv found"); return
    df = load_all(paths)
    tab = cells_table(df)
    outdir = "LLM_Blink/results"

    stats = [truncation_report(df), "\n## per model x load (report rates, blink, access)\n",
             tab.to_string(index=False)]
    with open(os.path.join(outdir, "scale_stats.txt"), "w") as f:
        f.write("\n".join(stats))
    print("\n".join(stats))

    fig3(tab, os.path.join(outdir, "fig3_blink_vs_size.png"))
    fig2(df, tab, os.path.join(outdir, "fig2_slopegraph.png"))
    fig6(df, os.path.join(outdir, "fig6_border_hists.png"))
    fig7(tab, df, os.path.join(outdir, "fig7_leaderboards.png"))
    print(f"\nfigures written to {outdir}/  ({len(paths)} models)")


if __name__ == "__main__":
    main()
