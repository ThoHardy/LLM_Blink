"""Forked-resampling runner (issues #14 + #18) — per-trial (s, k) counts.

Two modes, chosen by ``--resample-full``:

* **resample-full (issue #18 §2.3/§4, the campaign default):** ``--resample-full K``
  draws K *complete* trajectories per trial, regenerating the whole assistant turn
  from the empty prefix (fork offset 0), and scores BOTH borders on the SAME K
  samples — ``s_access/K`` (passphrase in ``<Thinking>``) and ``s_report/K``
  (passphrase in ``<Final_Answers>``). This is the graded per-trial visibility
  index of §2.3: it gives figures 2, 3, 6 and 7 from one pass. Per-sample vectors
  (``t1_correct``, realised CoT length, both border bits) are exported for the
  fig-2 violins and fig-6 histograms. Truncation is tracked per sample and the
  per-cell rate is printed (§2.1 gate: a cell over 2 % is a FAILED cell — re-run
  with a bigger ``--max-new-tokens``, never analyse it, never drop trials).

* **legacy mid-trajectory forks (issue #14):** ``--resample-full 0`` (the old
  default) forks a single shared trajectory at the read-out transition (report
  probe) and after ``<Thinking>`` (access probe). Kept for back-compat / the #14
  corpus; NOT what issue #18 measures (it holds the CoT fixed — pitfall #5).

Resumable & incremental as before: an existing out CSV's (load, regime,
passphrase_last, seed, base_temp) cells are skipped; each row is flushed at once.
Seed-major so partial results cover every condition early.

Usage (issue #18 Campaign A, run from the folder CONTAINING LLM_Blink/):
    python3 -B LLM_Blink/probe_pilot.py --model qwen2.5:3b \
        --loads trivial math_bench_2 math_bench_4 math_bench_5 \
        --regimes cot direct --n-seeds 100 \
        --resample-full 20 --report-forks 0 --access on \
        --base-temp 1.0 --budget none --max-new-tokens 4096 \
        --n-workers 8 --keep-logs \
        --out LLM_Blink/results/scale_qwen2.5_3b.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # parent so `import LLM_Blink` works

from LLM_Blink import load_model                                    # noqa: E402
from LLM_Blink.model import OllamaBackend                           # noqa: E402
from LLM_Blink.stimuli import TrialConfig, build_trial             # noqa: E402
from LLM_Blink.protocol import generate_trajectory                 # noqa: E402
from LLM_Blink.readout import (parse_task_report_names,            # noqa: E402
                               parse_task_report, task_rows,
                               answer_contains, t2_in_cot)
from LLM_Blink.resample import (fork_samples, score_report_samples,  # noqa: E402
                                score_access_samples, resample_full,
                                score_full_samples, DegenerateForkError)

# legacy (mid-trajectory fork) schema — issue #14
FIELDS = [
    "model", "n_tasks", "naming", "t1_load", "regime", "passphrase_last",
    "base_temp", "probe_k", "seed", "t2_phrase", "t2_task_name", "t2_rank",
    "t2_abs_index", "cot_len_chars", "base_output_truncated", "has_fork_point",
    "realized_report_contains", "realized_t2_in_cot", "t1_correct",
    "report_s", "report_k", "report_degenerate",
    "access_s", "access_k", "access_degenerate",
]

# resample-full schema — issue #18 §7.2 (per-trial vectors kept in list columns)
FIELDS_FULL = [
    "model", "quant_tag", "n_tasks", "naming", "t1_load", "regime",
    "passphrase_last", "report_order", "load_engagement", "base_temp",
    "max_new_tokens", "seed", "t2_phrase", "t2_task_name", "t2_rank",
    "t2_abs_index",
    # both borders, same k samples
    "k", "s_access", "s_report",
    # per-sample vectors (semicolon-joined; the fig-2/fig-6 plumbing)
    "report_bits", "access_bits", "t1_per", "cot_len_per",
    # cell-level summaries / covariates
    "t1_mean", "cot_len_mean",
    # quality gates (§2.1)
    "n_truncated_samples", "frac_truncated", "any_degenerate",
    # the realised base draw (for calibration / self-consistency)
    "base_cot_len_chars", "base_output_truncated",
    "realized_report_contains", "realized_t2_in_cot",
    # raw transcript kept for a subsample (§7.2)
    "raw_output",
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


def _quant_tag(model_name):
    """Ollama quant tag, verbatim (§2.2: refuse to plot a family on mixed tags).

    Queries the native /api/show endpoint; returns e.g. 'Q4_K_M' or '' on any
    failure (HF models, server down) so a missing tag never crashes a run.
    """
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"name": model_name}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.load(r)
        return (d.get("details", {}) or {}).get("quantization_level", "") or ""
    except Exception:
        return ""


def _realized_report(traj_text, task_name, phrase):
    parsed = parse_task_report_names(traj_text, [task_name])
    resp = parsed["reported"].get(str(task_name).upper())
    return answer_contains(resp, phrase)


def _t1_correct(traj_text, tr):
    """Load-task accuracy (fraction of load rows exact-correct), same formula
    as experiment.py:128 — panel 1 of the #10 dissociation figure."""
    parsed = parse_task_report(traj_text, tr)
    rows_t = task_rows(tr, parsed["reported"])
    load_rows = [r for r in rows_t if r["kind"] == "load"]
    if not load_rows:
        return ""
    return sum(r["correct"] for r in load_rows) / len(load_rows)


def _join(xs):
    return ";".join("" if x is None or x == "" else str(x) for x in xs)


def _mean(xs):
    nums = [float(x) for x in xs if x != "" and x is not None]
    return round(sum(nums) / len(nums), 4) if nums else ""


# ---------------------------------------------------------------------------
# resample-full mode (issue #18)
# ---------------------------------------------------------------------------

def run_full(args):
    model, tok = load_model(args.model)
    quant = _quant_tag(args.model) if isinstance(model, OllamaBackend) else ""
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _done_cells(out)
    new_file = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS_FULL)
    if new_file:
        w.writeheader(); f.flush()

    logs = None
    if args.keep_logs:
        logs = open(out.replace(".csv", "") + ".samples.jsonl", "a")

    budget = None if str(args.budget).lower() == "none" else int(args.budget)

    cells = [(load, regime, s) for s in range(args.n_seeds)
             for load in args.loads for regime in args.regimes]
    print(f"[probe_pilot:full] {args.model} (quant={quant!r}): {len(cells)} cells, "
          f"{len(done)} done, K={args.resample_full}, max_new_tokens={args.max_new_tokens}, "
          f"report_order={args.report_order}, load_engagement={args.load_engagement}",
          flush=True)

    # running per-cell truncation tally for the §2.1 gate
    trunc_num, trunc_den = {}, {}
    t0 = time.time(); n_run = 0
    for load, regime, s in cells:
        eff_load = load if args.n_tasks != 1 else "none"
        seed = _seed(args.n_tasks, eff_load, s)
        key = (eff_load, regime, str(args.passphrase_last), str(seed), str(args.base_temp))
        if key in done:
            continue

        cfg = TrialConfig(n_tasks=args.n_tasks, naming=args.naming,
                          t1_load=eff_load, regime=regime,
                          passphrase_last=args.passphrase_last,
                          anti_enumeration=args.anti_enumeration,
                          report_order=args.report_order,
                          load_engagement=args.load_engagement, seed=seed)
        tr = build_trial(cfg)

        # one base draw at base_temp: logs the realised trajectory + covariates
        base = generate_trajectory(model, tok, tr, finite_budget=budget,
                                   temperature=args.base_temp,
                                   max_new_tokens=args.max_new_tokens)

        # K full regenerations from offset 0 (the §2.3 measure)
        samples, truncs = resample_full(
            model, tok, tr, k=args.resample_full, temperature=1.0,
            seed=2 * seed + 1, max_new_tokens=args.max_new_tokens,
            n_workers=args.n_workers)
        score_access = (regime == "cot" and args.access == "on")
        sc = score_full_samples(samples, tr, score_access=score_access)

        n_trunc = int(sum(truncs))
        trunc_num[(eff_load, regime)] = trunc_num.get((eff_load, regime), 0) + n_trunc
        trunc_den[(eff_load, regime)] = trunc_den.get((eff_load, regime), 0) + len(truncs)

        keep_raw = (n_run < args.raw_subsample)
        row = dict(
            model=args.model, quant_tag=quant, n_tasks=args.n_tasks,
            naming=args.naming, t1_load=eff_load, regime=regime,
            passphrase_last=args.passphrase_last, report_order=args.report_order,
            load_engagement=args.load_engagement, base_temp=args.base_temp,
            max_new_tokens=args.max_new_tokens, seed=seed,
            t2_phrase=tr.t2_phrase, t2_task_name=tr.t2_task_name,
            t2_rank=tr.t2_rank, t2_abs_index=tr.t2_abs_index,
            k=sc["k"], s_access=("" if sc["s_access"] is None else sc["s_access"]),
            s_report=sc["s_report"],
            report_bits=_join(sc["report_bits"]),
            access_bits=("" if sc["access_bits"] is None else _join(sc["access_bits"])),
            t1_per=_join(sc["t1_per"]), cot_len_per=_join(sc["cot_len_per"]),
            t1_mean=_mean(sc["t1_per"]), cot_len_mean=_mean(sc["cot_len_per"]),
            n_truncated_samples=n_trunc,
            frac_truncated=round(n_trunc / len(truncs), 4) if truncs else "",
            any_degenerate=int(sc["degenerate"]),
            base_cot_len_chars=len(base.text),
            base_output_truncated=int(bool(base.output_truncated)),
            realized_report_contains=int(_realized_report(
                base.text, tr.t2_task_name, tr.t2_phrase)),
            realized_t2_in_cot=(int(t2_in_cot(base.text, tr.t2_phrase, tr.t2_task_name))
                                if regime == "cot" else ""),
            raw_output=(base.text if keep_raw else ""),
        )
        w.writerow(row); f.flush()
        if logs is not None:
            logs.write(json.dumps(dict(
                seed=seed, t1_load=eff_load, regime=regime,
                t2_phrase=tr.t2_phrase, t2_task_name=tr.t2_task_name,
                base=base.text, samples=samples, truncs=truncs)) + "\n")
            logs.flush()
        n_run += 1
        if n_run % 5 == 0:
            el = time.time() - t0
            gt = trunc_num.get((eff_load, regime), 0)
            gd = trunc_den.get((eff_load, regime), 1)
            print(f"  {n_run} trials, {el:.0f}s ({el/n_run:.1f}s/trial) "
                  f"[{eff_load}/{regime}] cell-trunc={100*gt/gd:.1f}%", flush=True)
    f.close()
    if logs is not None:
        logs.close()

    # §2.1 gate report at the end
    print("\n[probe_pilot:full] per-cell truncation (>2% = FAILED cell, re-run bigger):")
    worst = 0.0
    for (L, R), den in sorted(trunc_den.items()):
        frac = 100 * trunc_num[(L, R)] / den
        worst = max(worst, frac)
        flag = "  <-- FAILED (>2%)" if frac > 2.0 else ""
        print(f"    {L:16s} {R:7s} trunc={frac:5.2f}% (n_samples={den}){flag}")
    print(f"[probe_pilot:full] done: {n_run} new trials in {time.time()-t0:.0f}s; "
          f"worst-cell truncation {worst:.2f}%", flush=True)


# ---------------------------------------------------------------------------
# legacy mid-trajectory fork mode (issue #14) — unchanged behaviour
# ---------------------------------------------------------------------------

def run_legacy(args):
    model, tok = load_model(args.model)
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _done_cells(out)
    new_file = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); f.flush()

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
    p.add_argument("--report-order", dest="report_order", default="none",
                   choices=["none", "stream", "reverse"])
    p.add_argument("--load-engagement", dest="load_engagement", default="solve",
                   choices=["solve", "ignore"])
    p.add_argument("--n-seeds", type=int, default=150)
    p.add_argument("--k", type=int, default=20, help="legacy mid-trajectory fork k")
    p.add_argument("--resample-full", dest="resample_full", type=int, default=0,
                   help="K>0 -> #18 mode: K full regenerations/trial, both borders")
    p.add_argument("--report-forks", dest="report_forks", type=int, default=-1,
                   help="issue #18 §4: pass 0 to switch OFF the old mid-trajectory "
                        "forking (implied whenever --resample-full>0)")
    p.add_argument("--access", choices=["on", "off"], default="on")
    p.add_argument("--budget", default="none",
                   help="'none' (issue #18: uncapped CoT) or an int finite_budget")
    p.add_argument("--base-temp", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--keep-logs", dest="keep_logs", action="store_true",
                   help="write every base+K-sample transcript to <out>.samples.jsonl")
    p.add_argument("--raw-subsample", dest="raw_subsample", type=int, default=200,
                   help="keep raw_output in the CSV for the first N trials (§7.2)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.resample_full > 0:
        run_full(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
