"""Forked-resampling pilot runner (issue #14) — produce per-trial (s, k) counts.

Runs the two graded probes on a grid of cells and writes ONE CSV row per trial
with the counts the beta-binomial mixture (`mixture.py`) consumes. Standalone
(does not go through experiment.run_trial, which also computes the teacher-forced
graded log-prob that is NaN on Ollama) and deliberately lean, so a first real
answer to the all-or-none-vs-graded question lands fast and we can ITERATE.

Per trial, off ONE base trajectory:
  * base binary report_contains + t2_in_cot (the realized draw) — for the §5
    calibration/self-consistency checks;
  * report probe (fork at the read-out transition, k samples at T=1) -> report_s/k;
  * access probe (fork after <Thinking>, resample the whole CoT) -> access_s/k
    (cot regime only).

Resumable: an existing out CSV is read and its (load, regime, passphrase_last,
seed, base_temp) cells are skipped. Incremental: each row is flushed immediately,
so a kill / SSH drop never loses completed trials.

Seeds: the stimulus seed uses experiment.run_sweep's formula (comparable to the
existing corpus); each probe draws its k sampling seeds from a SEPARATE numpy
Generator (resample.sample_seeds), never from the stimulus RNG (7.2.2).

Usage (run from the folder CONTAINING LLM_Blink/):
    python LLM_Blink/probe_pilot.py --model gemma2:2b \
        --loads trivial semantic_2 semantic_4 --regimes cot direct \
        --n-tasks 5 --n-seeds 150 --k 20 --access on \
        --out LLM_Blink/results/probe_gemma2_2b.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # parent so `import LLM_Blink` works

from LLM_Blink import load_model                                    # noqa: E402
from LLM_Blink.stimuli import TrialConfig, build_trial             # noqa: E402
from LLM_Blink.protocol import generate_trajectory                 # noqa: E402
from LLM_Blink.readout import (parse_task_report_names,            # noqa: E402
                               parse_task_report, task_rows,
                               answer_contains, t2_in_cot)
from LLM_Blink.resample import (fork_samples, score_report_samples,  # noqa: E402
                                score_access_samples, DegenerateForkError)

FIELDS = [
    "model", "n_tasks", "naming", "t1_load", "regime", "passphrase_last",
    "base_temp", "probe_k", "seed", "t2_phrase", "t2_task_name", "t2_rank",
    "t2_abs_index", "cot_len_chars", "base_output_truncated", "has_fork_point",
    "realized_report_contains", "realized_t2_in_cot", "t1_correct",
    "report_s", "report_k", "report_degenerate",
    "access_s", "access_k", "access_degenerate",
]


def _seed(nt, load, s):
    # experiment.run_sweep formula (lag=0), so cells line up with the corpus.
    return 1000 * s + 0 + 7 * len(load) + 13 * (nt or 0)


def _done_cells(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["t1_load"], r["regime"], r["passphrase_last"],
                      r["seed"], r["base_temp"]))
    return done


def _realized_report(traj_text, task_name, phrase):
    parsed = parse_task_report_names(traj_text, [task_name])
    resp = parsed["reported"].get(str(task_name).upper())
    return answer_contains(resp, phrase)


def _t1_correct(traj_text, tr):
    """Load-task accuracy (fraction of load rows exact-correct), same formula
    as experiment.py:128 — panel 1 of the #10 dissociation figure (CoT should
    HELP this while it HURTS passphrase report)."""
    parsed = parse_task_report(traj_text, tr)
    rows_t = task_rows(tr, parsed["reported"])
    load_rows = [r for r in rows_t if r["kind"] == "load"]
    if not load_rows:
        return ""
    return sum(r["correct"] for r in load_rows) / len(load_rows)


def run(args):
    model, tok = load_model(args.model)
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _done_cells(out)
    new_file = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); f.flush()

    # seed-major ordering: partial results cover EVERY condition early (so the
    # mixture can be run on partial data while the rest accumulates).
    cells = []
    for s in range(args.n_seeds):
        for load in args.loads:
            for regime in args.regimes:
                cells.append((load, regime, s))
    print(f"[probe_pilot] {args.model}: {len(cells)} cells, "
          f"{len(done)} already done, k={args.k}, access={args.access}", flush=True)

    t0 = time.time()
    n_run = 0
    for i, (load, regime, s) in enumerate(cells):
        # n_tasks=1 collapses the load axis; only run load once there.
        eff_load = load if args.n_tasks != 1 else "none"
        seed = _seed(args.n_tasks, eff_load, s)
        key = (eff_load, regime, str(args.passphrase_last), str(seed), str(args.base_temp))
        if key in done:
            continue

        cfg = TrialConfig(n_tasks=args.n_tasks, naming=args.naming,
                          t1_load=eff_load, regime=regime,
                          passphrase_last=args.passphrase_last,
                          anti_enumeration=args.anti_enumeration, seed=seed)
        tr = build_trial(cfg)
        traj = generate_trajectory(model, tok, tr, finite_budget=None,
                                   temperature=args.base_temp,
                                   max_new_tokens=args.max_new_tokens)

        # report fork point: after the CoT for cot; the whole turn for direct
        # (direct output has no <Thinking>/<Final_Answers> scaffold to fork on).
        report_at = "post_cot" if regime == "cot" else "response"
        has_fork = traj.offset(report_at) is not None
        row = dict(
            model=args.model, n_tasks=args.n_tasks, naming=args.naming,
            t1_load=eff_load, regime=regime, passphrase_last=args.passphrase_last,
            base_temp=args.base_temp, probe_k=args.k, seed=seed,
            t2_phrase=tr.t2_phrase, t2_task_name=tr.t2_task_name,
            t2_rank=tr.t2_rank, t2_abs_index=tr.t2_abs_index,
            cot_len_chars=len(traj.text),
            base_output_truncated=int(bool(traj.output_truncated)),
            has_fork_point=int(has_fork),
            realized_report_contains=int(_realized_report(
                traj.text, tr.t2_task_name, tr.t2_phrase)),
            realized_t2_in_cot=int(t2_in_cot(traj.text, tr.t2_phrase, tr.t2_task_name))
                                if regime == "cot" else "",
            t1_correct=_t1_correct(traj.text, tr),
            report_s="", report_k="", report_degenerate="",
            access_s="", access_k="", access_degenerate="",
        )

        # report probe (every row with a fork point)
        if has_fork:
            try:
                reps = fork_samples(model, tok, tr, traj, at=report_at,
                                    k=args.k, temperature=1.0,
                                    seed=2 * seed + 1, n_workers=args.n_workers)
                deg = False
            except DegenerateForkError as e:
                reps, deg = e.samples, True
            sc = score_report_samples(reps, tr.t2_task_name, tr.t2_phrase)
            row.update(report_s=sc["s"], report_k=sc["k"],
                       report_degenerate=int(deg or sc["degenerate"]))

        # access probe (cot only, when enabled)
        if args.access == "on" and regime == "cot" and traj.offset("pre_cot") is not None:
            try:
                accs = fork_samples(model, tok, tr, traj, at="pre_cot",
                                    k=args.k, temperature=1.0,
                                    seed=2 * seed + 2, n_workers=args.n_workers)
                dega = False
            except DegenerateForkError as e:
                accs, dega = e.samples, True
            sca = score_access_samples(accs, tr.t2_phrase, tr.t2_task_name)
            row.update(access_s=sca["s"], access_k=sca["k"],
                       access_degenerate=int(dega or sca["degenerate"]))

        w.writerow(row); f.flush()
        n_run += 1
        if n_run % 10 == 0:
            el = time.time() - t0
            print(f"  {n_run} trials in {el:.0f}s ({el/n_run:.1f}s/trial) "
                  f"[{load}/{regime}]", flush=True)
    f.close()
    print(f"[probe_pilot] done: {n_run} new trials in {time.time()-t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--loads", nargs="+", default=["trivial", "semantic_2", "semantic_4"])
    p.add_argument("--regimes", nargs="+", default=["cot", "direct"])
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--naming", default="non-ordered")
    p.add_argument("--passphrase-last", dest="passphrase_last",
                   action="store_true", default=True)
    p.add_argument("--anti-enumeration", dest="anti_enumeration",
                   action="store_true", default=True)
    p.add_argument("--n-seeds", type=int, default=150)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--access", choices=["on", "off"], default="on")
    p.add_argument("--base-temp", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
