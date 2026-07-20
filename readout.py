"""Read-out layer (design H): parse the free report, locate scoring slots.

Handles the task-stream design's free report — one line per task in the form
``- Task <NAME>: <result>`` — and provides the teacher-forced scoring prefix
for the passphrase task. Legacy-template parsing ("Target 2 Result:") stays
in experiment.py.

Parsing is deliberately tolerant (missing dash, any case, stray quotes):
formatting slips must not masquerade as detection failures. The name-cued
probe (two-stage report -> full access taxonomy) is deliberately NOT
implemented yet (Thomas, 2026-07-20): for now a task is either spontaneously
reported or not.
"""
from __future__ import annotations
import re


_TASK_LINE_RE = re.compile(r"^\s*-?\s*Task\s+([A-Za-z0-9]+)\s*:\s*(.*?)\s*$",
                           re.IGNORECASE | re.MULTILINE)
_FINAL_OPEN = "<Final_Answers>"


def normalize_answer(text):
    """Uppercase, strip quotes/backticks and trailing punctuation, collapse
    whitespace. Applied to BOTH sides of every correctness comparison."""
    if text is None:
        return None
    t = str(text).strip().strip('"\N{APOSTROPHE}`').strip()
    t = re.sub(r"[.!?]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.upper()


def _answers_region(generated: str):
    """(region, offset) to parse for task lines: everything after
    <Final_Answers> when present — the CoT may legitimately restate task
    lines while reasoning — else the whole text."""
    i = generated.find(_FINAL_OPEN)
    if i == -1:
        return generated, 0
    off = i + len(_FINAL_OPEN)
    return generated[off:], off


def parse_task_report(generated: str, trial) -> dict:
    """Parse the free report.

    Returns ``{"reported": {NAME: raw_answer, ...}, "hallucinated": [...]}``
    where NAME is uppercased and only the FIRST line per task name counts.
    Hallucinated = well-formed task lines whose name matches no task in the
    trial (fabricated tasks — a detection-side error worth logging).
    """
    region, _ = _answers_region(generated)
    valid = {t["name"].upper() for t in trial.tasks}
    reported, hallucinated = {}, []
    for m in _TASK_LINE_RE.finditer(region):
        name = m.group(1).upper()
        if name in valid:
            reported.setdefault(name, m.group(2))
        else:
            hallucinated.append(name)
    return {"reported": reported, "hallucinated": hallucinated}


def task_rows(trial, reported: dict) -> list:
    """Per-task result dicts (JSON-ready) for the ``tasks`` column."""
    rows = []
    for t in trial.tasks:
        resp = reported.get(t["name"].upper())
        rows.append(dict(
            name=t["name"], kind=t["kind"], packet=t["packet"], rank=t["rank"],
            answer=t["answer"],
            reported=resp is not None,
            response=resp,
            correct=(normalize_answer(resp) == normalize_answer(t["answer"]))
                    if resp is not None else False,
        ))
    return rows


def t2_scoring_prefix_tasks(generated: str, task_name: str):
    """Split the generated text at the passphrase task's answer slot.

    Searches the <Final_Answers> region (NOT the CoT, which may re-enumerate
    packets and echo task names) for the model's own ``- Task <NAME>:`` line;
    the prefix ends right after the marker (+ optional opening quote), i.e.
    exactly where the model's own passphrase tokens would begin. If the model
    never emitted the slot, fall back to appending a fresh slot line to the
    full text and flag the trial (slot_missing=True) — same convention as the
    legacy ``_t2_scoring_prefix``.
    """
    region, off = _answers_region(generated)
    m = re.search(r"-?\s*Task\s+" + re.escape(task_name) + r"\s*:\s*\"?",
                  region, re.IGNORECASE)
    if m:
        return generated[: off + m.end()], False
    return generated.rstrip() + f"\n- Task {task_name}: ", True
