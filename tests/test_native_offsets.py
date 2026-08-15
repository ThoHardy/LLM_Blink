"""Fork-point re-anchoring for native reasoning models (issue #14, Step 7).

Qwen3 / DeepSeek-R1 etc. emit their own ``<think>...</think>`` block and then a
free-form answer, instead of our ``<Thinking>`` / ``<Final_Answers>`` scaffold.
The probes' fork points (protocol.Trajectory.offset) must re-anchor on the
native delimiters — and must NOT change behaviour on scaffold output. These
tests run offline (no model), so they gate the re-anchoring before any GPU run.

Run: ``python3 -B tests/test_native_offsets.py`` or via pytest.
"""
from __future__ import annotations
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO))
from LLM_Blink.protocol import Trajectory  # noqa: E402


def _mk(text):
    return Trajectory(text=text, output_truncated=False, finite_budget=None,
                      cot_tokens_used=None, cot_forced_closed=None)


def test_native_fork_points():
    t = _mk("<think>\nFind the passphrase. TANGO INDIA XRAY.\n</think>\n"
            "- Task FOO: TANGO INDIA XRAY")
    pre = t.offset("pre_cot")
    post = t.offset("post_cot")
    assert t.text[pre:pre + 4] == "Find", t.text[pre:pre + 4]
    assert t.text[post:post + 6] == "- Task", t.text[post:post + 6]


def test_scaffold_unchanged():
    t = _mk("<Thinking>\nreasoning\n</Thinking>\n"
            "<Final_Answers>\n- Task FOO: X\n</Final_Answers>")
    assert t.offset("pre_cot") == 11
    assert t.offset("post_cot") == 49
    assert t.offset("post_answers") == 79


def test_scaffold_jump_to_answers_unchanged():
    # gemma2:2b frequently skips </Thinking> and jumps to <Final_Answers>.
    t = _mk("<Thinking>\nreasoning\n<Final_Answers>\n- Task FOO: X")
    assert t.offset("post_cot") == 37


def test_no_thinking_returns_none():
    t = _mk("just some text with no reasoning block at all")
    assert t.offset("pre_cot") is None
    assert t.offset("post_cot") is None


if __name__ == "__main__":
    test_native_fork_points()
    test_scaffold_unchanged()
    test_scaffold_jump_to_answers_unchanged()
    test_no_thinking_returns_none()
    print("all native-offset tests pass")
