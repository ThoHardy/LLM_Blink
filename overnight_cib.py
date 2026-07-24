"""Overnight CoT-induced blindness (CIB) sweep — simplified design (issue #10).

Thomas' resimplified spec (2026-07-23):
  - k = 5 tasks (headline) + k = 1 (the "CoT normally helps" contrast)
  - finite_budget = None ONLY (unlimited CoT — the budget lever is dropped)
  - passphrase_last = True ONLY (T2 analog sits in the last rank)
  - ONE leaderboard per load: trivial, semantic_1..4 (no pooling)
  - many seeds (fresh stimulus per seed), temperature 0
  - metric of interest = Direct - CoT report_contains at k=5, per load
  - term used everywhere: "CoT-induced blindness" (not "blink")

Per model, the grid is:
  k=5 : {trivial, semantic_1..4} x {cot, direct} x n_seeds   (10 cells)
  k=1 : {none}                   x {cot, direct} x n_seeds   ( 2 cells)
The stimulus is identical across regimes for a given (k, load, seed) cell
(seed is regime-independent) so Direct-vs-CoT is PAIRED. Seed / cell formula
is copied from experiment.run_sweep so results match the sequential path
bit-for-bit at temperature 0.

Trials are independent HTTP calls -> dispatched across a thread pool (the
Ollama backend is thread-safe; the server runs OLLAMA_NUM_PARALLEL=8 so the
GPU batches concurrent decodes; see MACHINE.md). Models run small -> large,
one CSV per model in results/cib_<slug>.csv (raw_output untouched), resumable
(a model whose CSV already exists is skipped). One JSON line per model to
results/cib_progress.jsonl with a quick per-load CIB readout so partial
results stay decisive if the night ends early.

Usage:
    python overnight_cib.py                      # full ladder, n_seeds=100
    python overnight_cib.py --smoke              # 1 small model, n_seeds=5
    python overnight_cib.py --models qwen2.5:1.5b gemma2:2b --n-seeds 50
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # parent, so `import LLM_Blink` works

from LLM_Blink import load_model, backfill_readout_columns   # noqa: E402
from LLM_Blink.experiment import run_trial                    # noqa: E402
from LLM_Blink.stimuli import TrialConfig                     # noqa: E402
from LLM_Blink.model import DEFAULT_MAX_NEW_TOKENS            # noqa: E402

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
PROGRESS = os.path.join(RESULTS, "cib_progress.jsonl")

# Scale ladder, small -> large. Instruct-tuned models only: reasoning models
# (deepseek-r1, qwq, qwen3-thinking) emit their own CoT and would confound the
# cot/direct manipulation. (tag, params_B, family).
LADDER = [
    ("qwen2.5:0.5b",     0.5, "Qwen2.5"),
    ("qwen2.5:1.5b",     1.5, "Qwen2.5"),
    ("gemma2:2b",        2.0, "Gemma2"),
    ("llama3.2:3b",      3.0, "Llama3"),
    ("qwen2.5:3b",       3.0, "Qwen2.5"),
    ("gemma3:4b",        4.0, "Gemma3"),
    ("mistral:7b",       7.0, "Mistral"),
    ("qwen2.5:7b",       7.0, "Qwen2.5"),
    ("llama3.1:8b",      8.0, "Llama3"),
    ("gemma2:9b",        9.0, "Gemma2"),
    ("falcon3:10b",     10.0, "Falcon3"),
    ("gemma3:12b",      12.0, "Gemma3"),
    ("mistral-nemo:12b",12.0, "Mistral"),
    ("phi4:14b",        14.0, "Phi4"),
    ("qwen2.5:14b",     14.0, "Qwen2.5"),
    ("mistral-small:24b",24.0,"Mistral"),
    ("gemma2:27b",      27.0, "Gemma2"),
    ("gemma3:27b",      27.0, "Gemma3"),
    ("qwen2.5:32b",     32.0, "Qwen2.5"),
    ("llama3.3:70b",    70.0, "Llama3"),
    ("qwen2.5:72b",     72.0, "Qwen2.5"),
]

K5_LOADS = ("trivial", "semantic_1", "semantic_2", "semantic_3", "semantic_4")
REGIMES = ("cot", "direct")


def _slug(tag):
    return tag.replace("/", "_").replace(":", "_")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cfgs(n_seeds: int):
    """(n_tasks, load, regime, seed) cells + TrialConfigs, mirroring run_sweep.

    seed = 1000*s + lag + 7*len(load) + 13*n_tasks, lag = 0. Regime-independent
    -> cot and direct in the same (k, load, s) cell share the stimulus (paired).
    k=1 has no load tasks: load collapses to "none".
    """
    cfgs = []
    for nt in (1, 5):
        loads = K5_LOADS if nt > 1 else ("none",)
        for load, regime, s in itertools.product(loads, REGIMES, range(n_seeds)):
            seed = 1000 * s + 7 * len(load) + 13 * nt
            cfgs.append(TrialConfig(
                lag=0, t1_load=load, regime=regime, mask=False,
                n_pre=None, n_post=None,
                seed=seed, n_tasks=nt, naming="non-ordered",
                passphrase_last=True, anti_enumeration=True))
    return cfgs


def cib_summary(df: pd.DataFrame):
    """Per-load Direct - CoT report_contains at k=5 (gated on not-truncated).

    Also the k=1 contrast (single-task cot vs direct) and t1_correct under each
    regime (control 1: CoT should HELP the load task). Returns a plain dict.
    """
    g = df[df["output_truncated"] == False]
    out = {"k5_by_load": {}, "k1": {}}
    k5 = g[g["n_tasks"] == 5]
    for load in K5_LOADS:
        c = k5[(k5["t1_load"] == load) & (k5["regime"] == "cot")]
        d = k5[(k5["t1_load"] == load) & (k5["regime"] == "direct")]
        if len(c) and len(d):
            cot_rc, dir_rc = c["report_contains"].mean(), d["report_contains"].mean()
            out["k5_by_load"][load] = {
                "n_cot": int(len(c)), "n_direct": int(len(d)),
                "cot_report": round(float(cot_rc), 3),
                "direct_report": round(float(dir_rc), 3),
                "cib": round(float(dir_rc - cot_rc), 3),   # the headline metric
                "cot_t1": None if c["t1_correct"].isna().all()
                          else round(float(c["t1_correct"].mean()), 3),
                "direct_t1": None if d["t1_correct"].isna().all()
                             else round(float(d["t1_correct"].mean()), 3),
            }
    k1 = g[g["n_tasks"] == 1]
    c1, d1 = k1[k1["regime"] == "cot"], k1[k1["regime"] == "direct"]
    if len(c1) and len(d1):
        out["k1"] = {
            "cot_report": round(float(c1["report_contains"].mean()), 3),
            "direct_report": round(float(d1["report_contains"].mean()), 3),
            "cib": round(float(d1["report_contains"].mean()
                               - c1["report_contains"].mean()), 3),
        }
    return out


def run_model(tag, params, family, n_seeds, workers, max_new_tokens):
    t0 = time.time()
    out_csv = os.path.join(RESULTS, f"cib_{_slug(tag)}.csv")
    if os.path.exists(out_csv):
        log(f"skip {tag} (CSV exists: {os.path.basename(out_csv)})")
        return None
    log(f"=== {tag} ({params}B, {family}) — loading ===")
    model, tok = load_model(tag)
    cfgs = build_cfgs(n_seeds)
    n = len(cfgs)
    log(f"  {tag}: {n} trials ({n_seeds} seeds x 12 cells), {workers} workers")

    rows = [None] * n
    done = 0

    def work(i):
        return i, run_trial(model, tok, cfgs[i], temperature=0.0,
                            finite_budget=None, max_new_tokens=max_new_tokens)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in range(n)]
        for fut in as_completed(futs):
            try:
                i, row = fut.result()
                rows[i] = row
            except Exception:
                log(f"  !! trial failed:\n{traceback.format_exc()}")
            done += 1
            if done % 50 == 0 or done == n:
                el = time.time() - t0
                log(f"  {tag}: {done}/{n}  {el:.0f}s ({el/done:.2f}s/trial)")

    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    df["model"] = tag
    df["params_b"] = params
    df["family"] = family
    df = backfill_readout_columns(df)
    df.to_csv(out_csv, index=False)

    summ = cib_summary(df)
    rec = {
        "model": tag, "params_b": params, "family": family,
        "n_trials": int(len(df)), "n_seeds": n_seeds,
        "trunc_rate": round(float(df["output_truncated"].mean()), 3),
        "cib": summ, "elapsed_min": round((time.time() - t0) / 60, 1),
        "csv": os.path.basename(out_csv),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    # quick console readout
    s2 = summ["k5_by_load"].get("semantic_2", {})
    log(f"  DONE {tag} in {rec['elapsed_min']}min | sem2 k5: "
        f"direct={s2.get('direct_report')} cot={s2.get('cot_report')} "
        f"CIB={s2.get('cib')} | k1 CIB={summ['k1'].get('cib')} "
        f"| trunc={rec['trunc_rate']} -> {out_csv}")
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   help="override the ladder with explicit ollama tags")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--smoke", action="store_true",
                   help="1 small model, n_seeds=5 — validate + measure throughput")
    args = p.parse_args()

    if args.smoke:
        ladder = [("qwen2.5:1.5b", 1.5, "Qwen2.5")]
        n_seeds = 5
    elif args.models:
        by_tag = {t: (t, pb, fam) for t, pb, fam in LADDER}
        ladder = [by_tag.get(t, (t, 0.0, "?")) for t in args.models]
        n_seeds = args.n_seeds
    else:
        ladder = LADDER
        n_seeds = args.n_seeds

    log(f"CIB overnight sweep: {len(ladder)} model(s), n_seeds={n_seeds}, "
        f"workers={args.workers}, max_new_tokens={args.max_new_tokens}")
    for tag, params, family in ladder:
        try:
            run_model(tag, params, family, n_seeds, args.workers,
                      args.max_new_tokens)
        except Exception:
            log(f"  !! {tag} FAILED:\n{traceback.format_exc()}")
    log("CIB overnight sweep complete.")


if __name__ == "__main__":
    main()
