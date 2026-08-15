"""Overnight A x B x H budget sweep across the local Ollama model zoo.

Per model:
  1. ANCHOR   -> natural CoT length (median cot_tokens_used at a high,
                 non-binding budget of 1024, over non-forced-closed trials).
  2. BUDGETS  -> derived as ~{25%, 50%, 100%} of that median + None (unlimited),
                 because absolute token budgets are not comparable across scales;
                 relative scarcity is (ticket #10, Phase 1 step 1).
  3. COT GRID -> k in {1,3,5} x budgets x {trivial, hard_load} x both pp arms.
  4. DIRECT   -> the parallel-readout anchor (budget axis collapses to None).
  5. SAVE     -> backfilled CSV per model in results/, raw_output untouched.
  6. LOG      -> one JSON line per model to results/overnight_progress.jsonl,
                 plus a quick blink-score readout so partial results stay decisive.

Ordered small -> large so the scale ladder is decisive even if the night ends
early. Each model is wrapped in try/except: one failure never kills the run.
"""
import os, sys, json, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # parent, so `import LLM_Blink` works

from LLM_Blink import load_model, run_sweep, backfill_readout_columns  # noqa: E402
import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
PROGRESS = os.path.join(RESULTS, "overnight_progress.jsonl")

# (ollama_tag, params_B, family, hard_load) -- small -> large
MODELS = [
    ("qwen2.5:0.5b", 0.5, "Qwen2.5", "semantic_2"),
    ("qwen2.5:1.5b", 1.5, "Qwen2.5", "semantic_2"),
    ("gemma2:2b",    2.0, "Gemma2",  "semantic_2"),
    ("qwen2.5:3b",   3.0, "Qwen2.5", "semantic_3"),
    ("gemma3:4b",    4.0, "Gemma3",  "semantic_3"),
    ("mistral:7b",   7.0, "Mistral", "semantic_4"),
    ("qwen2.5:7b",   7.0, "Qwen2.5", "semantic_4"),
    ("gemma2:9b",    9.0, "Gemma2",  "semantic_4"),
    ("gemma3:12b",  12.0, "Gemma3",  "semantic_4"),
    ("mistral-nemo:12b", 12.0, "Mistral", "semantic_4"),
    ("qwen2.5:14b", 14.0, "Qwen2.5", "semantic_4"),
]

ANCHOR_SEEDS = 6
SEEDS        = 8
ANCHOR_CAP   = 1024      # high, non-binding budget to read natural CoT length
K_TASKS      = (1, 3, 5)
ARMS         = (False, True)   # random-rank (leaderboard) + last-rank (position stress)


def _slug(tag):
    return tag.replace("/", "_").replace(":", "_")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def derive_budgets(median):
    lv = sorted({max(8, int(round(median * f))) for f in (0.25, 0.5, 1.0)})
    return tuple(lv) + (None,)


def natural_cot_median(model, tok, hard_load):
    df = run_sweep(model, tok,
                   n_tasks_list=(3, 5), finite_budgets=(ANCHOR_CAP,),
                   loads=("trivial",), regimes=("cot",),
                   passphrase_last=(False,), n_seeds=ANCHOR_SEEDS,
                   verbose=False)
    used = df.loc[df["cot_forced_closed"] == False, "cot_tokens_used"].dropna()
    if len(used) == 0:                      # everything bound the cap -> verbose model
        used = df["cot_tokens_used"].dropna()
    return float(np.median(used)) if len(used) else float(ANCHOR_CAP), df


def blink_score(df, budgets):
    """v1 blink score: mean over binding relative budgets of
    (direct - cot(budget)) / direct, on report_contains, k>1, random-rank arm,
    both loads pooled. 0 = no serial penalty, 1 = collapse."""
    g = df[(df["regime"].isin(["cot", "direct"])) &
           (df["passphrase_last"] == False) &
           (df["n_tasks"] > 1) &
           (df["output_truncated"] == False)]
    direct = g[g["regime"] == "direct"]["report_contains"].mean()
    if not direct or np.isnan(direct):
        return None, direct
    finite = [b for b in budgets if b is not None]
    binding = sorted(finite)[:2]            # ~25% and ~50% = the binding levels
    ratios = []
    for b in binding:
        cot = g[(g["regime"] == "cot") & (g["finite_budget"] == b)]["report_contains"].mean()
        if cot is not None and not np.isnan(cot):
            ratios.append((direct - cot) / direct)
    score = float(np.mean(ratios)) if ratios else None
    return score, float(direct)


def run_model(tag, params, family, hard_load):
    t0 = time.time()
    log(f"=== {tag} ({params}B, {family}) ===")
    model, tok = load_model(tag)

    median, anchor_df = natural_cot_median(model, tok, hard_load)
    budgets = derive_budgets(median)
    log(f"  natural CoT median = {median:.0f} tok -> budgets {budgets}")

    cot = run_sweep(model, tok,
                    n_tasks_list=K_TASKS, finite_budgets=budgets,
                    loads=("trivial", hard_load), regimes=("cot",),
                    passphrase_last=ARMS, n_seeds=SEEDS, verbose=False)
    direct = run_sweep(model, tok,
                       n_tasks_list=K_TASKS, finite_budgets=(None,),
                       loads=("trivial", hard_load), regimes=("direct",),
                       passphrase_last=ARMS, n_seeds=SEEDS, verbose=False)

    df = pd.concat([anchor_df, cot, direct], ignore_index=True)
    df["model"] = tag
    df["params_b"] = params
    df["family"] = family
    df["natural_cot_median"] = median
    df = backfill_readout_columns(df)

    out = os.path.join(RESULTS, f"ab_{_slug(tag)}.csv")
    df.to_csv(out, index=False)

    score, direct_ceiling = blink_score(df, budgets)
    fc = df[df["cot_forced_closed"].notna()]["cot_forced_closed"].mean()
    rec = {
        "model": tag, "params_b": params, "family": family,
        "n_trials": int(len(df)),
        "natural_cot_median": round(median, 1),
        "budgets": [b for b in budgets if b is not None],
        "direct_ceiling_report": None if direct_ceiling is None else round(direct_ceiling, 3),
        "blink_score_v1": None if score is None else round(score, 3),
        "forced_close_rate": None if np.isnan(fc) else round(float(fc), 3),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "csv": os.path.basename(out),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"  DONE {tag}: blink_v1={rec['blink_score_v1']} "
        f"direct_ceiling={rec['direct_ceiling_report']} "
        f"forced_close={rec['forced_close_rate']} in {rec['elapsed_min']}min -> {out}")
    return rec


def main():
    log(f"Overnight sweep starting: {len(MODELS)} models, "
        f"anchor_seeds={ANCHOR_SEEDS} seeds={SEEDS} k={K_TASKS} arms=both")
    done = set()
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    for tag, params, family, hard_load in MODELS:
        if tag in done:
            log(f"skip {tag} (already in progress log)")
            continue
        try:
            run_model(tag, params, family, hard_load)
        except Exception:
            log(f"  !! {tag} FAILED:\n{traceback.format_exc()}")
    log("Overnight sweep complete.")


if __name__ == "__main__":
    main()
