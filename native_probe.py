"""Step 7 live validation: same-weights native reasoning vs non-reasoning (Qwen3).

The CoT-induced blindness must not be an artefact of OUR `<Thinking>` scaffold.
Test: take a loaded stimulus with the DIRECT output format (no `<Thinking>`
instruction) and toggle ONLY the model's own native reasoning via Ollama's
`think` flag — same weights, same prompt, one bit different:

  * native_think  (think=True):  Qwen3 reasons in its own <think> block, then
    writes the <Final_Answers> report -> the "reasoning" arm.
  * native_direct (think=False): Qwen3 answers with no reasoning -> baseline.

Prediction (if the blink is a genuine property of the reasoning stage, not our
scaffold): passphrase report is LOWER under native_think than native_direct.

Also reconstructs the inline `<think>...</think>{content}` trajectory from
Ollama's separated fields and runs the re-anchored report fork (protocol Step 7)
on it, to validate the offset code end-to-end on real native output.

Usage: python3 LLM_Blink/native_probe.py --model qwen3.5:35b-a3b --n 20
"""
from __future__ import annotations
import argparse
import json
import sys
import os
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from LLM_Blink.stimuli import TrialConfig, build_trial            # noqa: E402
from LLM_Blink.readout import parse_task_report_names, answer_contains  # noqa: E402
from LLM_Blink.protocol import Trajectory                          # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"


def _chat(model, system, user, think, num_predict=6144):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "think": think,
        "options": {"temperature": 1.0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d.get("message", {})
    return (m.get("content", "") or "", m.get("thinking", "") or "",
            d.get("done_reason"))


def _reported(content, task_name, phrase):
    parsed = parse_task_report_names(content, [task_name])
    resp = parsed["reported"].get(str(task_name).upper())
    return answer_contains(resp, phrase)


def _n_report_lines(content):
    return content.count("- Task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:35b-a3b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--load", default="semantic_4")
    ap.add_argument("--budget", type=int, default=6144)
    a = ap.parse_args()

    # Count passphrase report ONLY among COMPLETED reports (report emitted, not
    # truncated) so an empty/cut-off report from budget exhaustion is not
    # mistaken for a blink (that would be the external-interruption confound the
    # plan rejects). Truncation rate is reported separately.
    rep = {"think": 0, "direct": 0}         # passphrase reported | completed
    done = {"think": 0, "direct": 0}        # completed reports
    trunc = {"think": 0, "direct": 0}       # truncated / empty report
    fork_ok = 0
    for s in range(a.n):
        seed = 1000 * s + 7 * len(a.load) + 13 * 5
        cfg = TrialConfig(n_tasks=5, naming="non-ordered", t1_load=a.load,
                          regime="direct", passphrase_last=True, seed=seed)
        tr = build_trial(cfg)
        for think, key in ((True, "think"), (False, "direct")):
            content, thinking, dr = _chat(a.model, tr.system, tr.user, think,
                                          num_predict=a.budget)
            # a completed report = emitted the report (>=3 of 5 task lines) and
            # not budget-truncated.
            completed = (dr != "length") and (_n_report_lines(content) >= 3)
            if not completed:
                trunc[key] += 1
            else:
                done[key] += 1
                if _reported(content, tr.t2_task_name, tr.t2_phrase):
                    rep[key] += 1
            if think:
                inline = f"<think>\n{thinking}\n</think>\n{content}"
                traj = Trajectory(text=inline, output_truncated=False,
                                  finite_budget=None, cot_tokens_used=None,
                                  cot_forced_closed=None)
                if (traj.offset("pre_cot") is not None
                        and traj.offset("post_cot") is not None):
                    fork_ok += 1
        print(f"  {s+1}/{a.n}: think rep {rep['think']}/{done['think']} "
              f"(trunc {trunc['think']}) | direct rep {rep['direct']}/{done['direct']} "
              f"(trunc {trunc['direct']}) | fork_ok {fork_ok}", flush=True)

    def rate(k):
        return rep[k] / done[k] if done[k] else float("nan")
    print(f"\n=== {a.model}, load={a.load}, n={a.n}, budget={a.budget} ===")
    print(f"native_direct (no reasoning): passphrase reported "
          f"{rep['direct']}/{done['direct']} completed = {rate('direct'):.2f} "
          f"({trunc['direct']} truncated)")
    print(f"native_think  (reasoning)   : passphrase reported "
          f"{rep['think']}/{done['think']} completed = {rate('think'):.2f} "
          f"({trunc['think']} truncated)")
    print(f"blink among completed reports (direct - think) = "
          f"{rate('direct') - rate('think'):+.2f}")
    print(f"native fork points re-anchored OK on <think>/</think>: "
          f"{fork_ok}/{a.n}")


if __name__ == "__main__":
    main()
