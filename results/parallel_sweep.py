"""Parallel AB sweep driver — same trials as run_sweep, run concurrently.

Reuses load_model + run_trial from the package unchanged. Trials are fully
independent, so we dispatch the identical (lag, load, regime, mask, n_pre,
n_post, seed) cells across a thread pool. Threads are the right tool here: the
Ollama backend is pure HTTP I/O (OpenAI/httpx client, thread-safe, shared),
so the GIL is released during each request and Ollama batches the concurrent
decodes on the GPU (needs OLLAMA_NUM_PARALLEL >= workers to actually batch).

CRITICAL: the seed formula and cell-product order are copied verbatim from
experiment.run_sweep so a parallel run is bit-identical to the sequential one
(temperature=0). If run_sweep changes, mirror it here.

Usage:
    python parallel_sweep.py --model gemma2:9b --workers 8 \
        --output results/ab_gemma2_9b_t0_s10.csv --n-seeds 10
"""
import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# import the package (results/ is one level under LLM_Blink/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                 # .../LLM_Blink
_ROOT = os.path.dirname(_PKG)                 # parent of LLM_Blink
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

from LLM_Blink import load_model            # noqa: E402
from LLM_Blink.experiment import run_trial  # noqa: E402
from LLM_Blink.stimuli import TrialConfig   # noqa: E402


def build_cells(lags, loads, regimes, masks, n_pres, n_posts, n_seeds,
                include_example=True, t2_words=3):
    """Exact copy of run_sweep's cell + seed construction."""
    cells = list(itertools.product(lags, loads, regimes, masks, n_pres, n_posts,
                                   range(n_seeds)))
    cfgs = []
    for (lag, load, regime, mask, n_pre, n_post_, seed) in cells:
        cfgs.append(TrialConfig(
            lag=lag, t1_load=load, regime=regime, mask=mask,
            n_pre=n_pre, n_post=n_post_,
            include_example=include_example, t2_words=t2_words,
            seed=1000 * seed + lag + 7 * len(load)))
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ollama-url", default="http://localhost:11434/v1")
    p.add_argument("--lags", nargs="+", type=int, default=[0, 2, 4, 6, 8, 10])
    p.add_argument("--loads", nargs="+", default=["none", "semantic_4"])
    p.add_argument("--regimes", nargs="+", default=["cot", "direct"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--no-example", action="store_true",
                   help="Drop the fixed worked example from the prompt (2026-06-05 "
                        "pre-example condition; restores headroom in T2 report).")
    p.add_argument("--t2-words", type=int, default=3,
                   help="Length of the T2 passphrase (titration: longer = harder "
                        "to report, brings binary report off ceiling).")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cfgs = build_cells(args.lags, tuple(args.loads), tuple(args.regimes),
                       (False,), (None,), (None,), args.n_seeds,
                       include_example=not args.no_example, t2_words=args.t2_words)
    n = len(cfgs)
    print(f"Model   : {args.model}")
    print(f"Workers : {args.workers}")
    print(f"Trials  : {n}  (temp={args.temperature}, max_tok={args.max_new_tokens})")
    print(f"Output  : {args.output}")

    print("Loading model...")
    model, tok = load_model(args.model, ollama_base_url=args.ollama_url)
    print("Model ready.\n")

    rows = [None] * n
    done = 0
    t0 = time.time()

    def work(i):
        return i, run_trial(model, tok, cfgs[i],
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(n)]
        for fut in as_completed(futs):
            i, row = fut.result()
            rows[i] = row
            done += 1
            if done % 10 == 0 or done == n:
                el = time.time() - t0
                print(f"  {done}/{n}  {el:.0f}s  ({el/done:.2f}s/trial, "
                      f"{done/el:.2f} trial/s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    el = time.time() - t0
    print(f"\nSaved {len(df)} rows to {args.output} in {el:.0f}s "
          f"({el/n:.2f}s/trial wall).")


if __name__ == "__main__":
    main()
