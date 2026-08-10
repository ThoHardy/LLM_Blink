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
finite_budget=0    -> REMOVED (2026-07-21, Thomas): the zero point of the
                      budget axis IS the direct regime. A forced-empty
                      '<Thinking>\n</Thinking>' is an ambiguous stimulus (the
                      template announces reasoning that never happens) and may
                      confuse the model for no reason — use regime='direct'
                      for the no-consolidation condition instead. Passing 0
                      raises ValueError.
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

``continue_generate`` works on BOTH backends (HF and, since PR #11 / 2026-07-22,
Ollama via the native /api/chat prefill path) — the forked-resampling probes of
issue #14 rely on it on Ollama.
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

    # -- fork points (issue #14, Step 1) -----------------------------------
    # Character offsets into ``text`` for the forked-resampling probes. A fork
    # at offset O rebuilds ``prompt + text[:O]`` and resamples the continuation.
    #
    # The read-out transition is defined by the ANSWER-BLOCK OPENER
    # ``<Final_Answers>``, not by ``</Thinking>``: small non-ceiling models
    # (gemma2:2b) frequently jump straight from reasoning into <Final_Answers>
    # WITHOUT closing </Thinking> (</Thinking> present in only ~19% of gemma2:2b
    # cot trials vs <Final_Answers> in ~99%). Forking on </Thinking> alone would
    # leave the primary probe undefined on 80% of the very trials that carry the
    # effect. ``<Final_Answers>`` gives ~99% fork coverage on every model tried.
    _ANSWERS_OPEN = "<Final_Answers>"

    def offset(self, at: str) -> int | None:
        """Char offset into ``text`` of a fork point, or None if absent.

        at="pre_cot"   -> just after the ``<Thinking>`` opener (resample the
                          whole CoT; the access probe). Stops at the answer opener.
        at="post_cot"  -> just after the read-out transition = the ``<Final_Answers>``
                          opener if present, else after ``</Thinking>`` (resample
                          the report; the primary report probe).
        at="post_answers" -> just after ``</Final_Answers>`` (end of report).
        """
        t = self.text
        if at == "pre_cot":
            i = t.find(THINKING_OPEN.rstrip("\n"))    # "<Thinking>"
            if i == -1:
                return None
            o = i + len(THINKING_OPEN.rstrip("\n"))
            if o < len(t) and t[o] == "\n":
                o += 1
            return o
        if at == "post_cot":
            i = t.find(self._ANSWERS_OPEN)            # read-out transition
            if i != -1:
                o = i + len(self._ANSWERS_OPEN)
                if o < len(t) and t[o] == "\n":
                    o += 1
                return o
            j = t.find(STOP_THINKING)                 # fallback: clean template
            return j + len(STOP_THINKING) if j != -1 else None
        if at == "post_answers":
            i = t.find(STOP_ANSWERS)
            return i + len(STOP_ANSWERS) if i != -1 else None
        raise ValueError(f"unknown fork point {at!r}")


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

    if finite_budget == 0:
        raise ValueError(
            "finite_budget=0 was removed (2026-07-21): an empty forced "
            "'<Thinking></Thinking>' is an ambiguous stimulus. The zero point "
            "of the budget axis is regime='direct' — use that instead.")
    if not 0 < finite_budget <= MAX_FINITE_BUDGET:
        raise ValueError(
            f"finite_budget={finite_budget} out of range (0, {MAX_FINITE_BUDGET}]")
    if not trial.forced_close_text:
        raise ValueError(
            "trial.forced_close_text is empty: cannot force-close a capped "
            "<Thinking> block (was the trial built for regime='cot'?)")

    # -- stage 1: the rationed <Thinking> span -------------------------------
    cot_text, used, _s1_trunc = continue_generate(
        model, tok, trial.system, trial.user,
        prefilled_assistant=THINKING_OPEN,
        max_new_tokens=finite_budget, temperature=temperature,
        stop_at=STOP_THINKING)
    if STOP_THINKING in cot_text:
        forced = False
        prefix = THINKING_OPEN + cot_text
    else:                              # cap hit mid-thought: that's the manipulation
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
