"""Generation protocol (design A): finite CoT budget with forced close.

``finite_budget`` caps ONLY the tokens generated inside the ``<Thinking>``
block — between the opening and closing tags (Thomas's spec, 2026-07-20).
When the cap is hit before the model closes the block on its own,
``Trial.forced_close_text`` is injected and the answer stage continues with
its own (generous) ``answer_budget``, so the REPORT channel is never cut by
the budget: consolidation time is the only thing rationed. This turns the
2026-07-17 truncation artifact into the instrument, by construction.

Semantics
---------
finite_budget=None -> one generation pass, exactly the legacy behavior.
finite_budget=0    -> forced-empty <Thinking> (no stage-1 generation at all);
                      bridges toward the 'direct' regime / encoding baseline.
finite_budget=n>0  -> stage 1: continue from '<Thinking>\\n' with
                      max_new_tokens=n, stopping at '</Thinking>'. If the stop
                      never fires, the block is force-closed
                      (cot_forced_closed=True). Stage 2: continue from the
                      realized prefix with ``answer_budget``, stopping at
                      '</Final_Answers>'.

Only meaningful for regime='cot': for 'direct' there is no <Thinking> to
ration, so the budget is ignored (Trajectory.finite_budget logs None — the
APPLIED value). Requires the HuggingFace backend when a budget is applied
(``continue_generate`` is HF-only).

Because stage 1 starts from the prefilled opener, the <Thinking> block is
guaranteed open by construction, and ``cot_tokens_used`` counts pure
Thinking-span tokens (including the closing tag when the model closed
naturally).
"""
from __future__ import annotations
from dataclasses import dataclass

from .model import report_generate, continue_generate, DEFAULT_MAX_NEW_TOKENS

MAX_FINITE_BUDGET = 2000
DEFAULT_ANSWER_BUDGET = 512
THINKING_OPEN = "<Thinking>\n"
STOP_THINKING = "</Thinking>"
STOP_ANSWERS = "</Final_Answers>"


@dataclass
class Trajectory:
    """Realized assistant trajectory + protocol flags (one per trial)."""
    text: str                       # full realized text (prefills included)
    output_truncated: bool          # answer stage / single pass hit its budget
    finite_budget: int | None       # budget actually APPLIED (None = single pass)
    cot_tokens_used: int | None     # stage-1 generated tokens (None = single pass)
    cot_forced_closed: bool | None  # True = cap hit, close injected


def generate_trajectory(model, tok, trial, finite_budget: int | None = None,
                        temperature: float = 0.0,
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                        answer_budget: int = DEFAULT_ANSWER_BUDGET) -> Trajectory:
    """Generate one trial's full assistant trajectory (see module docstring)."""
    if finite_budget is not None and trial.config.regime != "cot":
        finite_budget = None          # the budget is a CoT knob

    if finite_budget is None:
        out, truncated = report_generate(
            model, tok, trial.system, trial.user,
            max_new_tokens=max_new_tokens, temperature=temperature,
            stop_at=STOP_ANSWERS, return_truncated=True)
        return Trajectory(out, truncated, None, None, None)

    if not 0 <= finite_budget <= MAX_FINITE_BUDGET:
        raise ValueError(
            f"finite_budget={finite_budget} out of range [0, {MAX_FINITE_BUDGET}]")
    if not trial.forced_close_text:
        raise ValueError(
            "trial.forced_close_text is empty: cannot force-close a capped "
            "<Thinking> block (was the trial built for regime='cot'?)")

    # -- stage 1: the rationed <Thinking> span -------------------------------
    if finite_budget == 0:
        prefix = THINKING_OPEN + trial.forced_close_text
        used, forced = 0, True
    else:
        cot_text, used, _s1_trunc = continue_generate(
            model, tok, trial.system, trial.user,
            prefilled_assistant=THINKING_OPEN,
            max_new_tokens=finite_budget, temperature=temperature,
            stop_at=STOP_THINKING)
        if STOP_THINKING in cot_text:
            forced = False
            prefix = THINKING_OPEN + cot_text
        else:                          # cap hit mid-thought: that's the manipulation
            forced = True
            sep = "" if (cot_text == "" or cot_text.endswith("\n")) else "\n"
            prefix = THINKING_OPEN + cot_text + sep + trial.forced_close_text

    # -- stage 2: answers, never rationed ------------------------------------
    ans_text, _n2, s2_trunc = continue_generate(
        model, tok, trial.system, trial.user,
        prefilled_assistant=prefix,
        max_new_tokens=answer_budget, temperature=temperature,
        stop_at=STOP_ANSWERS)
    return Trajectory(prefix + ans_text, s2_trunc, finite_budget, used, forced)
