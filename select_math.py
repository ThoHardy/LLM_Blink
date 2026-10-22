"""Selection pilot (issue #14, Thomas's MATH ask): find model x MATH-level cells
where CoT genuinely HELPS the load task (cot t1 > direct t1) — the precondition
for a clean 'CoT helps the load yet blinds the passphrase' paradox.

Lean by design: only the BASE trajectory per trial (no k-fork), so it scans a
grid ~1+k times faster than probe_pilot. Records t1_correct (load accuracy),
realized passphrase report, and realized in-CoT, so we also preview the blink.

Resumable/incremental like probe_pilot. Run from the folder CONTAINING LLM_Blink/:
    python LLM_Blink/select_math.py --model qwen2.5:7b \
        --loads math_bench_1 math_bench_2 math_bench_3 --regimes cot direct \
        --n-tasks 5 --n-seeds 15 --out LLM_Blink/results/select_qwen2.5_7b.csv
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
from LLM_Blink.readout import t2_in_cot                            # noqa: E402
from LLM_Blink.probe_pilot import _seed, _realized_report, _t1_correct  # noqa: E402

FIELDS = ["model", "n_tasks", "t1_load", "regime", "seed", "cot_len_chars",
          "base_output_truncated", "realized_report_contains",
          "realized_t2_in_cot", "t1_correct"]


def _done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["t1_load"], r["regime"], r["seed"]))
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
    cells = [(load, reg, s) for s in range(a.n_seeds)
             for load in a.loads for reg in a.regimes]
    print(f"[select] {a.model}: {len(cells)} cells, {len(done)} done", flush=True)
    t0 = time.time(); n = 0
    for load, reg, s in cells:
        eff = load if a.n_tasks != 1 else "none"
        seed = _seed(a.n_tasks, eff, s)
        if (eff, reg, str(seed)) in done:
            continue
        cfg = TrialConfig(n_tasks=a.n_tasks, naming=a.naming, t1_load=eff,
                          regime=reg, passphrase_last=True,
                          anti_enumeration=True, seed=seed)
        tr = build_trial(cfg)
        traj = generate_trajectory(model, tok, tr, finite_budget=None,
                                   temperature=a.base_temp,
                                   max_new_tokens=a.max_new_tokens)
        w.writerow(dict(
            model=a.model, n_tasks=a.n_tasks, t1_load=eff, regime=reg, seed=seed,
            cot_len_chars=len(traj.text),
            base_output_truncated=int(bool(traj.output_truncated)),
            realized_report_contains=int(_realized_report(
                traj.text, tr.t2_task_name, tr.t2_phrase)),
            realized_t2_in_cot=int(t2_in_cot(traj.text, tr.t2_phrase, tr.t2_task_name))
                                if reg == "cot" else "",
            t1_correct=_t1_correct(traj.text, tr)))
        f.flush(); n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n} trials {el:.0f}s ({el/n:.1f}s/trial) [{load}/{reg}]", flush=True)
    f.close()
    print(f"[select] done {n} new in {time.time()-t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--loads", nargs="+",
                   default=[f"math_bench_{i}" for i in range(1, 6)])
    p.add_argument("--regimes", nargs="+", default=["cot", "direct"])
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--naming", default="non-ordered")
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument("--base-temp", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
