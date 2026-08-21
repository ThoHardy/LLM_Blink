"""score_full_samples: both borders on the SAME k samples (issue #18 §2.3/§7.2).

Offline (no model): builds a real trial, crafts synthetic full-trajectory strings
whose <Thinking>/<Final_Answers> content is known, and checks that s_access and
s_report count the right samples, that the per-sample bits pair correctly, and
that the direct path (score_access=False) logs no access — the plumbing Campaign
A depends on. Run: ``python3 -B -m pytest tests/test_resample_full.py``.
"""
from __future__ import annotations
import os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO))
from LLM_Blink.stimuli import TrialConfig, build_trial          # noqa: E402
from LLM_Blink.resample import score_full_samples               # noqa: E402


def _trial():
    return build_trial(TrialConfig(n_tasks=4, naming="non-ordered",
                                   t1_load="trivial", regime="cot",
                                   passphrase_last=True, seed=7))


def _sample(think, answers):
    return f"<Thinking>\n{think}\n<Final_Answers>\n{answers}\n</Final_Answers>"


def test_both_borders_same_samples():
    tr = _trial()
    name, phrase = tr.t2_task_name, tr.t2_phrase
    samples = [
        # in CoT (by phrase) AND reported -> access 1, report 1
        _sample(f"Task {name}: I copy {phrase}.", f"- Task {name}: {phrase}"),
        # NOT in CoT but reported (the second route) -> access 0, report 1
        _sample("Only load tasks here, nothing else.", f"- Task {name}: {phrase}"),
        # in CoT (by name) but NOT reported -> access 1, report 0
        _sample(f"Task {name} is a copy task.", "- Task OTHER: nope"),
        # neither -> access 0, report 0
        _sample("Load tasks only.", "- Task OTHER: nope"),
    ]
    sc = score_full_samples(samples, tr, score_access=True)
    assert sc["k"] == 4
    assert sc["report_bits"] == [1, 1, 0, 0], sc["report_bits"]
    assert sc["access_bits"] == [1, 0, 1, 0], sc["access_bits"]
    assert sc["s_report"] == 2 and sc["s_access"] == 2
    # per-sample pairing is preserved (fig-1 decomposition)
    pairs = list(zip(sc["access_bits"], sc["report_bits"]))
    assert (0, 1) in pairs      # a non-verbalised report exists in this set


def test_direct_logs_no_access():
    tr = _trial()
    name, phrase = tr.t2_task_name, tr.t2_phrase
    # direct: no <Thinking>; access must be None, and must NOT match the phrase
    # that appears only in the report line.
    samples = [f"- Task {name}: {phrase}", "- Task OTHER: nope"]
    sc = score_full_samples(samples, tr, score_access=False)
    assert sc["s_access"] is None and sc["access_bits"] is None
    assert sc["s_report"] == 1 and sc["report_bits"] == [1, 0]


if __name__ == "__main__":
    test_both_borders_same_samples()
    test_direct_logs_no_access()
    print("ok")
