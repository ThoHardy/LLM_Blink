"""§3.1 matcher audit — read-only pass over saved raw_output.

Recomputes the CURRENT access matcher (readout.t2_in_cot: passphrase phrase OR
task-name, word-bounded, inside <Thinking>) on every cot-regime trial that kept
its transcript, and isolates the informative cell of issue #18 §3.1:

    matcher codes the passphrase ABSENT from the CoT, yet it IS reported.

For those trials it dumps the verbatim <Thinking> span so a human (me) can judge
whether the passphrase task is referred to in a way the matcher misses (named by
task-name variant, "the copy task", "the three words", "repeat at the end",
different casing/punctuation, spelled out, handled implicitly...).

Read-only. No GPU, no model. Run from the folder CONTAINING LLM_Blink/.
"""
from __future__ import annotations
import csv, glob, os, sys, json, argparse, random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from LLM_Blink.readout import (t2_in_cot, _thinking_region,
                               parse_task_report_names, answer_contains)

csv.field_size_limit(10 ** 7)


def reported_passphrase(raw, task_name, phrase):
    parsed = parse_task_report_names(raw, [task_name])
    resp = parsed["reported"].get(str(task_name).upper())
    return answer_contains(resp, phrase)


def scan(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("regime") != "cot":
                    continue
                raw = r.get("raw_output") or ""
                if not raw:
                    continue
                phrase = r.get("t2_phrase") or ""
                name = r.get("t2_task_name") or ""
                in_cot = t2_in_cot(raw, phrase, name)
                rep = reported_passphrase(raw, name, phrase)
                rows.append(dict(
                    model=r.get("model") or os.path.basename(path),
                    load=r.get("t1_load"), seed=r.get("seed"),
                    name=name, phrase=phrase,
                    in_cot=in_cot, reported=rep,
                    think=_thinking_region(raw),
                    raw=raw))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="LLM_Blink/results/cib_*.csv")
    ap.add_argument("--dump", type=int, default=40,
                    help="how many informative-cell transcripts to dump")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    rows = scan(paths)
    n = len(rows)
    absent = [r for r in rows if not r["in_cot"]]
    informative = [r for r in absent if r["reported"]]      # absent-but-reported

    print(f"files: {len(paths)}  cot-trials-with-transcript: {n}")
    print(f"matcher codes passphrase ABSENT from CoT: {len(absent)} "
          f"({100*len(absent)/n:.1f}%)")
    print(f"  of those, passphrase nonetheless REPORTED: {len(informative)} "
          f"({100*len(informative)/n:.1f}% of all; "
          f"{100*len(informative)/max(1,len(absent)):.1f}% of the absent set)")

    # by load
    print("\nabsent-but-reported rate by load:")
    loads = sorted({r["load"] for r in rows})
    for L in loads:
        sub = [r for r in rows if r["load"] == L]
        info = [r for r in sub if not r["in_cot"] and r["reported"]]
        print(f"  {L:14s} n={len(sub):5d}  absent&reported="
              f"{100*len(info)/max(1,len(sub)):5.1f}%")

    # by model
    print("\nabsent-but-reported rate by model:")
    models = sorted({r["model"] for r in rows})
    for M in models:
        sub = [r for r in rows if r["model"] == M]
        info = [r for r in sub if not r["in_cot"] and r["reported"]]
        print(f"  {M:22s} n={len(sub):5d}  absent&reported="
              f"{100*len(info)/max(1,len(sub)):5.1f}%")

    # dump a spread of transcripts for eyeballing
    rnd = random.Random(42)
    sample = informative[:]
    rnd.shuffle(sample)
    sample = sample[:args.dump]
    out = args.out or "LLM_Blink/results/_matcher_audit_dump.txt"
    with open(out, "w") as f:
        for i, r in enumerate(sample):
            f.write(f"\n{'='*78}\n[{i}] model={r['model']} load={r['load']} "
                    f"seed={r['seed']}\n")
            f.write(f"TASK NAME: {r['name']}   PASSPHRASE: {r['phrase']}\n")
            f.write(f"{'-'*78}\n<Thinking> span:\n{r['think'].strip()}\n")
            # also the reported line
            parsed = parse_task_report_names(r["raw"], [r["name"]])
            resp = parsed["reported"].get(str(r["name"]).upper())
            f.write(f"{'-'*78}\nreported answer for {r['name']}: {resp!r}\n")
    print(f"\ndumped {len(sample)} transcripts -> {out}")


if __name__ == "__main__":
    main()
