"""Nested probe runner (issue #14 §3.3/§4.3) — the clean report-stage test.

The report probe forks the ONE realized CoT, so its across-trial bimodality
partly inherits the CoT's content. The nested probe removes that confound: for
each of k_cot sampled CoTs it draws k_rep reports, giving the variance
decomposition (§4.3):

  ICC = between-CoT var / (between + within-CoT var)
  ICC -> 1  : report is near-deterministic GIVEN a CoT  -> access decided in the
              workspace = ignition (all-or-none).
  ICC << 1 : report is stochastic given a fixed CoT     -> graded read-out.

Writes ONE row per (trial, sampled CoT): the (s, k_rep) report count for that
CoT and whether the passphrase task was in it. Feed to nested_analyze / the
mixture's icc_nested. Resumable and incremental like probe_pilot.py.

Usage:
    python LLM_Blink/nested_pilot.py --model gemma2:2b \
        --loads trivial semantic_4 --n-seeds 40 --k-cot 6 --k-rep 6 \
        --out LLM_Blink/results/nested_gemma2_2b.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from LLM_Blink import load_model                                    # noqa: E402
from LLM_Blink.stimuli import TrialConfig, build_trial             # noqa: E402
from LLM_Blink.protocol import generate_trajectory                 # noqa: E402
from LLM_Blink.resample import nested_fork                         # noqa: E402

FIELDS = ["model", "t1_load", "regime", "seed", "t2_task_name", "t2_rank",
          "cot_id", "s", "k", "t2_in_this_cot"]


def _seed(nt, load, s):
    return 1000 * s + 0 + 7 * len(load) + 13 * (nt or 0)


def _done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["t1_load"], r["seed"]))
    return done


def run(a):
    model, tok = load_model(a.model)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = _done(a.out)
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader(); f.flush()

    cells = [(load, s) for s in range(a.n_seeds) for load in a.loads]
    print(f"[nested] {a.model}: {len(cells)} trials, {len(done)} done, "
          f"k_cot={a.k_cot} k_rep={a.k_rep}", flush=True)
    t0 = time.time(); n = 0
    for load, s in cells:
        seed = _seed(a.n_tasks, load, s)
        if (load, str(seed)) in done:
            continue
        cfg = TrialConfig(n_tasks=a.n_tasks, naming="non-ordered", t1_load=load,
                          regime="cot", passphrase_last=True,
                          anti_enumeration=True, seed=seed)
        tr = build_trial(cfg)
        traj = generate_trajectory(model, tok, tr, finite_budget=None,
                                   temperature=a.base_temp, max_new_tokens=1024)
        try:
            rows = nested_fork(model, tok, tr, traj, k_cot=a.k_cot,
                               k_rep=a.k_rep, temperature=1.0,
                               seed=seed, n_workers=a.n_workers)
        except Exception as e:
            print(f"  skip seed={seed} load={load}: {e}", flush=True)
            continue
        for r in rows:
            w.writerow(dict(model=a.model, t1_load=load, regime="cot", seed=seed,
                            t2_task_name=tr.t2_task_name, t2_rank=tr.t2_rank,
                            cot_id=r["cot_id"], s=r["s"], k=r["k"],
                            t2_in_this_cot=r["t2_in_this_cot"]))
        f.flush(); n += 1
        if n % 5 == 0:
            el = time.time() - t0
            print(f"  {n} trials in {el:.0f}s ({el/n:.1f}s/trial)", flush=True)
    f.close()
    print(f"[nested] done: {n} new trials in {time.time()-t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--loads", nargs="+", default=["trivial", "semantic_4"])
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--k-cot", type=int, default=6)
    p.add_argument("--k-rep", type=int, default=6)
    p.add_argument("--base-temp", type=float, default=1.0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
