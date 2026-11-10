"""Issue #21 Step 1 summary for gemma3:4b: Thomas's table + truncation gate + drift check.
Run from the folder containing LLM_Blink/."""
import glob, re
import pandas as pd

SLUG = "gemma3_4b"
R = "LLM_Blink/results"
files = {"campaignA_base": f"{R}/scale_{SLUG}.csv"}
files.update({re.match(rf".*/cond_(.+)_{SLUG}\.csv", p).group(1): p
              for p in glob.glob(f"{R}/cond_*_{SLUG}.csv")})
rows = []
for cond, path in sorted(files.items()):
    d = pd.read_csv(path)
    d["p"] = d.s_report / d.k
    for load, g in list(d.groupby("t1_load")) + [("ALL", d)]:
        m = g.groupby("regime").p.mean()
        tau = g.loc[g.regime == "cot", "order_tau_mean"].mean() if "order_tau_mean" in g else float("nan")
        rows.append(dict(condition=cond, load=load, n_stimuli=int((g.regime == "cot").sum()),
                         direct=m.get("direct"), cot=m.get("cot"),
                         rel_blink=(m.get("direct") - m.get("cot")) / m.get("direct"),
                         access_cot=(g.loc[g.regime == "cot", "s_access"] / g.k).mean(),
                         t1_cot=g.loc[g.regime == "cot", "t1_mean"].mean(),
                         order_tau_cot=tau))
out = pd.DataFrame(rows).round(3)
print(out[out.load == "ALL"].drop(columns="load").to_string(index=False))
out.to_csv(f"{R}/summary_conditions_{SLUG}.csv", index=False)

print("\n# Truncation gate (cell = condition x load x regime; fail if > 2%)")
for cond, path in sorted(files.items()):
    d = pd.read_csv(path)
    t = d.groupby(["t1_load", "regime"]).frac_truncated.mean()
    bad = t[t > 0.02]
    print(f"{cond:24s} worst={t.max():.4f}  {'FAIL ' + str(dict(bad.round(4))) if len(bad) else 'ok'}")

base_p = f"{R}/cond_base_{SLUG}.csv"
if glob.glob(base_p):
    print("\n# Drift check: cond_base vs scale on the same seeds (per load x regime)")
    key = ["t1_load", "regime", "seed"]
    b = pd.read_csv(base_p); a = pd.read_csv(files["campaignA_base"])
    for d in (a, b):
        d["p_rep"] = d.s_report / d.k; d["p_acc"] = d.s_access / d.k
    j = b.merge(a, on=key, suffixes=("_new", "_old"))
    g = j.groupby(["t1_load", "regime"]).agg(n=("seed", "size"),
        report_old=("p_rep_old", "mean"), report_new=("p_rep_new", "mean"),
        access_old=("p_acc_old", "mean"), access_new=("p_acc_new", "mean"),
        t1_old=("t1_mean_old", "mean"), t1_new=("t1_mean_new", "mean"))
    print(g.round(3).to_string())
    print("same base prompt (raw_output differs by sampling only):",
          j.t2_phrase_new.eq(j.t2_phrase_old).mean() if "t2_phrase_new" in j else "n/a")
