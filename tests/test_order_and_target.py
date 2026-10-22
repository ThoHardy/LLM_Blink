"""Compute-order / example-order / target-kind axes (2026-10-21).

Offline. Checks byte-exact defaults, exact pairing across target kinds, the
prompt and worked-example text of each arm, the access/report scoring of the
new targets, the processing-order manipulation check, and the probe_pilot
resume key.
Run: ``python3 -B -m pytest tests/test_order_and_target.py``.
"""
from __future__ import annotations
import os, sys, hashlib, re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO))
from LLM_Blink.stimuli import TrialConfig, build_trial               # noqa: E402
from LLM_Blink.readout import (t2_in_cot, access_cues, answer_contains,  # noqa: E402
                               compute_order_respected,
                               passphrase_processing_rank)

DEFAULT_DIGEST = "05af816af638d0d23aaefb17c4c6ab0b18e1e6b81fa445995928beeb96c5f234"


def _cfg(**kw):
    base = dict(n_tasks=5, naming="non-ordered", t1_load="math_bench_4",
                regime="cot", seed=11)
    base.update(kw)
    return TrialConfig(**base)


def test_defaults_byte_exact():
    h = hashlib.sha256()
    for load in ["trivial", "math_bench_4"]:
        for reg in ["cot", "direct"]:
            for seed in range(3):
                tr = build_trial(TrialConfig(n_tasks=5, naming="non-ordered",
                                             t1_load=load, regime=reg, seed=seed))
                h.update((tr.user + tr.t2_phrase).encode())
    assert h.hexdigest() == DEFAULT_DIGEST


def _non_target_lines(tr):
    pk = tr.t2_abs_index
    stream = tr.user.split("DATA STREAM:\n")[-1]
    return [l for l in stream.splitlines() if not l.startswith(f"[Packet {pk:02d}")]


@pytest.mark.parametrize("kind", ["digits", "trivial_math"])
@pytest.mark.parametrize("seed", range(20))
def test_target_kinds_are_paired(kind, seed):
    a = build_trial(_cfg(seed=seed))
    b = build_trial(_cfg(seed=seed, target_kind=kind))
    assert a.t2_abs_index == b.t2_abs_index and a.t2_task_name == b.t2_task_name
    assert [t for t in a.tasks if t["kind"] == "load"] == \
           [t for t in b.tasks if t["kind"] == "load"]
    assert _non_target_lines(a) == _non_target_lines(b)
    load_answers = {str(t["answer"]).upper() for t in b.tasks if t["kind"] == "load"}
    if kind == "digits":
        nums = b.t2_phrase.split()
        assert len(nums) == 3 and all(n.isdigit() and len(n) == 2 for n in nums)
        assert not load_answers.intersection(nums)
        assert f'Copy-paste these three numbers: "{b.t2_phrase}".' in b.user
    else:
        assert b.t2_phrase not in load_answers
        a_, b_ = map(int, re.findall(r"What is (\d) \+ (\d)\?", b.user)[-1])
        assert str(a_ + b_) == b.t2_phrase


def test_worked_example_matches_target_kind():
    assert 'Copy-paste these three numbers: "38 71 25".' in \
        build_trial(_cfg(target_kind="digits")).user
    u = build_trial(_cfg(target_kind="trivial_math")).user
    assert "What is 2 + 6?" in u and "- Task MARMALADE: 8" in u


def test_trivial_math_access_cue_is_the_expression():
    tr = build_trial(_cfg(target_kind="trivial_math"))
    expr = re.findall(r"What is (\d \+ \d)\?", tr.user)[-1]
    assert access_cues(tr) == (expr, expr.replace(" ", ""))
    cot = f"<Thinking>Lots of math: {tr.t2_phrase} apples.</Thinking>"
    assert not t2_in_cot(cot, access_cues(tr))
    assert t2_in_cot(f"<Thinking>{expr} is easy</Thinking>", access_cues(tr))
    assert answer_contains(f"{expr} = {tr.t2_phrase}", tr.t2_phrase)


def test_compute_order_sentences():
    base = build_trial(_cfg()).user
    for order, frag in [("stream", "in the order in which they appeared"),
                        ("reverse", "in reverse order of their appearance"),
                        ("load_first", "all other tasks before handling any copy-paste")]:
        u = build_trial(_cfg(compute_order=order)).user
        assert frag in u and frag not in base
        assert u.split("EXAMPLE")[0].rstrip().endswith("stream." if order != "load_first"
                                                       else "task.")


def _example(u):
    return u.split("OUTPUT:\n")[1].split("Now process")[0]


def test_example_order_and_compute_order_reorder_the_example():
    d = _example(build_trial(_cfg()).user)
    assert d.index("Task CATHEDRAL: a salmon") < d.index("Task MARMALADE: I must copy")
    pf = build_trial(_cfg(example_order="passphrase_first")).user
    assert '[Packet 02 - Task CATHEDRAL]: Copy-paste these three words' in pf
    e = _example(pf)
    assert e.index("Task CATHEDRAL: I must copy") < e.index("Task MARMALADE: a salmon")
    lf = _example(build_trial(_cfg(example_order="passphrase_first",
                                   compute_order="load_first")).user)
    assert lf.index("Task MARMALADE: a salmon") < lf.index("Task CATHEDRAL: I must copy")
    assert lf.index("- Task MARMALADE: ANIMAL") < lf.index("- Task CATHEDRAL: VICTOR")
    rv = _example(build_trial(_cfg(compute_order="reverse", regime="direct")).user)
    assert rv.index("- Task MARMALADE") < rv.index("- Task CATHEDRAL")


def test_prompt_only_axes_do_not_touch_the_stream():
    a = build_trial(_cfg(passphrase_rank=1))
    for kw in [dict(compute_order="load_first"), dict(example_order="passphrase_first")]:
        b = build_trial(_cfg(passphrase_rank=1, **kw))
        assert a.tasks == b.tasks and a.t2_phrase == b.t2_phrase
        assert a.user.split("DATA STREAM:\n")[-1].split("OUTPUT FORMAT")[0] == \
               b.user.split("DATA STREAM:\n")[-1].split("OUTPUT FORMAT")[0]


@pytest.mark.parametrize("kw", [
    dict(compute_order="bogus"), dict(example_order="bogus"), dict(target_kind="bogus"),
    dict(target_kind="trivial_math", compute_order="load_first"),
    dict(target_kind="trivial_math", load_engagement="ignore"),
])
def test_invalid_combinations_raise(kw):
    with pytest.raises(ValueError):
        build_trial(_cfg(**kw))


def _cot_text(tr, order_names):
    by = {t["name"]: t for t in tr.tasks}
    think = " ".join(f"Task {n}: result {by[n]['answer']}." for n in order_names)
    ans = "\n".join(f"- Task {t['name']}: {t['answer']}" for t in tr.tasks)
    return f"<Thinking>\n{think}\n</Thinking>\n<Final_Answers>\n{ans}\n</Final_Answers>"


def test_processing_order_check():
    tr = build_trial(_cfg(passphrase_rank=1, compute_order="load_first"))
    stream = [t["name"] for t in sorted(tr.tasks, key=lambda t: t["rank"])]
    pp = tr.t2_task_name
    loads_then_pp = [n for n in stream if n != pp] + [pp]
    assert compute_order_respected(_cot_text(tr, loads_then_pp), tr, "load_first") == 1.0
    assert compute_order_respected(_cot_text(tr, stream), tr, "load_first") < 0
    assert passphrase_processing_rank(_cot_text(tr, loads_then_pp), tr) == 5
    assert passphrase_processing_rank(_cot_text(tr, stream), tr) == 1
    no_pp = _cot_text(tr, [n for n in stream if n != pp])
    assert passphrase_processing_rank(no_pp, tr) is None
    assert compute_order_respected(_cot_text(tr, stream[::-1]), tr, "reverse") == 1.0
    assert compute_order_respected(_cot_text(tr, stream), tr, "none") == 1.0


def test_processing_order_ordered_naming_ignores_bare_digits():
    tr = build_trial(_cfg(naming="ordered"))
    text = ("<Thinking>\n2 + 3 = 5 and 4 * 1 = 4. Task 2: done. Task 1: done.\n"
            "</Thinking>\n<Final_Answers>\n</Final_Answers>")
    assert passphrase_processing_rank(text, tr) is None
    assert compute_order_respected(text, tr, "none") == -1.0


def test_resume_key():
    from LLM_Blink.probe_pilot import _cell_key
    old_row = dict(n_tasks="5", naming="non-ordered", t1_load="trivial", regime="cot",
                   passphrase_last="True", seed="7", base_temp="1.0")
    new_default = dict(n_tasks=5, naming="non-ordered", t1_load="trivial", regime="cot",
                       passphrase_spec="last", anti_enumeration=True,
                       compute_order="none", example_order="load_first",
                       target_kind="nato", load_engagement="solve", seed=7, base_temp=1.0)
    assert _cell_key(old_row) == _cell_key(new_default)
    keys = {_cell_key(dict(new_default, **kw)) for kw in [
        {}, dict(passphrase_spec="1"), dict(compute_order="load_first"),
        dict(example_order="passphrase_first"), dict(target_kind="digits"),
        dict(target_kind="trivial_math")]}
    assert len(keys) == 6


def test_attention_cue():
    cue = "Make sure to answer ALL the tasks."
    a, b = build_trial(_cfg()), build_trial(_cfg(attention_cue=True))
    assert cue not in a.user and cue in b.user
    assert b.user.split("\nEXAMPLE")[0].endswith(cue)
    assert a.tasks == b.tasks
    assert b.user.replace(" " + cue, "", 1) == a.user


def test_resume_key_cue_and_think():
    from LLM_Blink.probe_pilot import _cell_key
    base = dict(n_tasks=5, naming="non-ordered", t1_load="trivial", regime="cot",
                passphrase_last="True", seed="7", base_temp="1.0")
    assert _cell_key(base) == _cell_key(dict(base, attention_cue=False, think="default"))
    assert len({_cell_key(base), _cell_key(dict(base, attention_cue=True)),
                _cell_key(dict(base, think="off"))}) == 3


def test_ollama_think_payload(monkeypatch):
    import json
    from LLM_Blink import model as M
    pytest.importorskip("openai")
    m = M.OllamaBackend("qwen3:4b")
    sent = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"message": {"content": "x</Final_Answers>"},
                                           "done_reason": "stop", "eval_count": 3}).encode()

    def fake_urlopen(req, *a, **k):
        sent.update(json.loads(req.data.decode()))
        return R()
    monkeypatch.setattr(M.urllib.request, "urlopen", fake_urlopen)
    m.continue_generate("s", "u", "", 10)
    assert "think" not in sent
    m.think = False
    text, trunc = m.generate("s", "u", 10, stop_at="</Final_Answers>", return_truncated=True)
    assert sent["think"] is False and text == "x" and not trunc
