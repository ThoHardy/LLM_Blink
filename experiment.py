"""Run the sweep and collect both read-outs per trial.

Combined A x B x H design (2026-07-20) — the DEFAULT:
- B: ``n_tasks`` tasks per stream (1 passphrase + n_tasks-1 loads), 15 packets.
- A: ``finite_budget`` caps the tokens generated inside <Thinking> only; on
  cap the block is force-closed and answers finish uncut (protocol.py).
- H: per-trial task names (``naming`` = "ordered" | "non-ordered"), hidden
  cardinality, free report "- Task <NAME>: <result>" (readout.py).
The legacy single-T1 lag design is still available via n_tasks=None
(run_sweep(n_tasks_list=(None,), finite_budgets=(None,), ...) or the CLI
--legacy flag) and stays BIT-EXACT for rescore_graded.py.

Read-outs, both off the SAME generated trajectory ("generate once, then
read", 2026-07-16):
- report_correct (bool): the passphrase task's reported answer equals the
  correct passphrase -> binary "conscious" report.
- t2_total_logprob / t2_joint_prob: teacher-forced joint prob of the CORRECT
  passphrase conditioned on the model's OWN generated prefix at the
  passphrase answer slot -> graded "unconscious strength". At temperature>0
  the prefix is the actually-sampled one (Thomas, 2026-07-16).
- t1_correct: legacy = bool (the single T1); task design = FRACTION of load
  tasks answered correctly (None when there are no load tasks).
- tasks (JSON, task design only): per-task dicts with reported/correct.

The old empty-CoT teacher-forced score survives as an OPTIONAL legacy-only
control via ``encoding_baseline=True`` (``*_encoding`` columns).
"""
from __future__ import annotations
import json
import re
import itertools
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # keep package dependency-light
    def tqdm(iterable, **kwargs):
        return iterable

from .stimuli import TrialConfig, build_trial
from .model import sequence_logprob, DEFAULT_MAX_NEW_TOKENS, OllamaBackend
from .protocol import generate_trajectory, DEFAULT_ANSWER_BUDGET
from .readout import (parse_task_report, task_rows,
                      t2_scoring_prefix_tasks, cot_enumeration_stats)


# ---------------------------------------------------------------------------
# Legacy-template helpers (numbered "Target N Result:" design). Kept verbatim:
# rescore_graded.py imports _t2_scoring_prefix, and legacy rows still use them.
# ---------------------------------------------------------------------------

def _extract_t2(text: str) -> str | None:
    m = re.search(r"Target 2 Result:\s*\"?([A-Z ]+?)\"?\s*(?:\n|$)", text)
    return m.group(1).strip() if m else None


def _extract_t1(text: str) -> str | None:
    # answers may be numbers (math loads) or short uppercase words (semantic loads)
    m = re.search(r"Target 1 Result:\s*\"?([^\"\n]+?)\"?\s*(?:\n|$)", text)
    return m.group(1).strip().upper() if m else None


# Text the legacy cot OUTPUT TEMPLATE uses as its <Thinking> placeholder. If it
# shows up verbatim in the output, the model copied the template instead of
# reasoning.
_PLACEHOLDER_RE = re.compile(r"\[process the stream")

# Task-stream analog: the OUTPUT FORMAT instruction copied into <Thinking>
# instead of actual reasoning (the phrase below appears nowhere else).
_PLACEHOLDER_TASKS_RE = re.compile(r"identify every task packet")

# The T2 slot marker in legacy generated output.
_T2_SLOT_RE = re.compile(r"Target 2 Result:\s*\"?")


def _t2_scoring_prefix(generated: str) -> tuple[str, bool]:
    """Split legacy generated text at the T2 slot for teacher-forced scoring.

    Returns (prefix, slot_missing); on a missing/malformed marker, falls back
    to appending the marker to the full generated text and flags the trial.
    """
    m = _T2_SLOT_RE.search(generated)
    if m:
        return generated[: m.end()], False
    return generated.rstrip() + "\nTarget 2 Result: ", True


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_trial(model, tok, cfg: TrialConfig, temperature: float = 0.0,
              encoding_baseline: bool = False,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
              finite_budget: int | None = None,
              answer_budget: int = DEFAULT_ANSWER_BUDGET) -> dict:
    tr = build_trial(cfg)
    legacy = cfg.n_tasks is None

    # -- 1. generation (single pass, or two-stage under a finite budget) ----
    traj = generate_trajectory(model, tok, tr, finite_budget=finite_budget,
                               temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               answer_budget=answer_budget)
    out = traj.text

    # -- 2. parse the report ------------------------------------------------
    if legacy:
        got_t2 = _extract_t2(out)
        got_t1 = _extract_t1(out)
        report_correct = (got_t2 == tr.t2_phrase)
        t1_correct = (got_t1 == tr.t1_answer) if tr.t1_answer else None
        prefix, slot_missing = _t2_scoring_prefix(out)
        thinking_placeholder = (bool(_PLACEHOLDER_RE.search(out))
                                if cfg.regime == "cot" else None)
        tasks_json = task_positions = None
        n_reported = n_halluc = None
        report_contains = answer_migration = None
        enum_stats = {"n_packets_in_cot": None,
                      "n_filler_packets_in_cot": None,
                      "enumerated_fillers_in_cot": None}
    else:
        parsed = parse_task_report(out, tr)
        rows_t = task_rows(tr, parsed["reported"])
        pp_row = next(r for r in rows_t if r["kind"] == "passphrase")
        report_correct = bool(pp_row["correct"])
        load_rows = [r for r in rows_t if r["kind"] == "load"]
        t1_correct = (sum(r["correct"] for r in load_rows) / len(load_rows)
                      if load_rows else None)
        prefix, slot_missing = t2_scoring_prefix_tasks(out, tr.t2_task_name)
        thinking_placeholder = (bool(_PLACEHOLDER_TASKS_RE.search(out))
                                if cfg.regime == "cot" else None)
        tasks_json = json.dumps(rows_t)
        task_positions = json.dumps([t["packet"] for t in tr.tasks])
        n_reported = len(parsed["reported"])
        n_halluc = len(set(parsed["hallucinated"]))
        # Lenient access + binding errors (2026-07-21, pilot review):
        report_contains = bool(pp_row["correct_lenient"])
        rep_rows = [r for r in rows_t if r["reported"]]
        answer_migration = (sum(r["answer_migrated"] for r in rep_rows)
                            / len(rep_rows)) if rep_rows else None
        # Anti-enumeration compliance (idea I1, 2026-07-22): distinct packet
        # indices referenced in <Thinking>; filler refs = non-compliance.
        enum_stats = (cot_enumeration_stats(out, [t["packet"] for t in tr.tasks])
                      if cfg.regime == "cot" else
                      {"n_packets_in_cot": None,
                       "n_filler_packets_in_cot": None,
                       "enumerated_fillers_in_cot": None})

    # -- 3. graded measure: P(correct passphrase | model's own prefix) ------
    # Rehearsal confound gate (2026-07-20): if the passphrase already appears
    # in the model's own pre-slot prefix (CoT echo / re-enumeration), the
    # teacher-forced score at the slot is near-copy probability. Stratify.
    t2_echoed = (cfg.regime == "cot") and (tr.t2_phrase in prefix.upper())
    lp = sequence_logprob(
        model, tok,
        system=tr.system,
        user_prefix=tr.user,
        target_text=tr.t2_phrase,
        prefilled_assistant=prefix,
    )

    row = {
        # design axes (A x B x H)
        "n_tasks": cfg.n_tasks,
        "naming": None if legacy else cfg.naming,
        "passphrase_last": None if legacy else cfg.passphrase_last,
        "anti_enumeration": None if legacy else cfg.anti_enumeration,
        "finite_budget": traj.finite_budget,   # APPLIED value (None = single pass)
        # legacy geometry (None on task-design rows)
        "lag": cfg.lag if legacy else None,
        "mask": cfg.mask if legacy else None,
        "n_pre": tr.n_pre_used, "n_post": tr.n_post_used,
        # shared condition columns
        "t1_load": cfg.t1_load, "regime": cfg.regime,
        "t2_abs_index": tr.t2_abs_index,
        "t2_rank": tr.t2_rank, "t2_task_name": tr.t2_task_name,
        "task_positions": task_positions,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "answer_budget": answer_budget if traj.finite_budget is not None else None,
        "seed": cfg.seed, "t2_words": cfg.t2_words,
        "t2_phrase": tr.t2_phrase,
        # graded read-out (off the model's own trajectory)
        "t2_total_logprob": lp["total_logprob"],
        "t2_mean_logprob": lp["mean_logprob"],
        "t2_joint_prob": lp["joint_prob"],
        "t2_n_tokens": lp["n_tokens"],
        # binary read-out (same generation pass)
        "report_correct": report_correct,
        # lenient access measures (2026-07-21): exact-match report_correct
        # undercounts access when the model wraps the right answer in echo
        # text. phrase_anywhere = passphrase appears ANYWHERE in the raw
        # output (incl. CoT) — the project's operational definition of
        # conscious access, at its most permissive.
        "report_contains": report_contains,
        "phrase_anywhere": tr.t2_phrase.upper() in out.upper(),
        "t1_correct": t1_correct,
        # per-task detail (task design only)
        "tasks": tasks_json,
        "n_tasks_reported": n_reported,
        "n_hallucinated_tasks": n_halluc,
        # fraction of REPORTED task lines carrying another task's answer
        # (binding error / illusory-conjunction analog; None if none reported)
        "answer_migration": answer_migration,
        # protocol flags (design A)
        "cot_tokens_used": traj.cot_tokens_used,
        "cot_forced_closed": traj.cot_forced_closed,
        # data-quality gates
        "output_truncated": traj.output_truncated,
        "t2_slot_missing": slot_missing,
        "thinking_is_placeholder": thinking_placeholder,
        "t2_echoed_in_cot": t2_echoed if cfg.regime == "cot" else None,
        # anti-enumeration compliance (task-design cot rows only)
        "n_packets_in_cot": enum_stats["n_packets_in_cot"],
        "n_filler_packets_in_cot": enum_stats["n_filler_packets_in_cot"],
        "enumerated_fillers_in_cot": enum_stats["enumerated_fillers_in_cot"],
        "raw_output": out,
    }

    # -- 4. optional legacy-only control: empty-CoT teacher-forced score ----
    if encoding_baseline:
        if not legacy:
            raise ValueError(
                "encoding_baseline is a legacy-design control (n_tasks=None); "
                "the task-stream template has no fixed answer-slot prefix.")
        lp0 = sequence_logprob(
            model, tok,
            system=tr.system,
            user_prefix=tr.user,
            target_text=tr.t2_phrase,
            prefilled_assistant=tr.template_prefix_after_prompt,
        )
        row["t2_total_logprob_encoding"] = lp0["total_logprob"]
        row["t2_mean_logprob_encoding"] = lp0["mean_logprob"]
        row["t2_joint_prob_encoding"] = lp0["joint_prob"]

    return row


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(model, tok,
              naming: str = "non-ordered",
              n_tasks_list=(1, 3, 7),
              finite_budgets=(64, 256, 1024),
              loads=("trivial", "semantic_4"),
              regimes=("cot",),
              passphrase_last=(True, False),
              anti_enumeration: bool = True,
              lags=(0,), masks=(False,),
              n_pres: tuple[int | None, ...] = (None,),
              n_posts: tuple[int | None, ...] = (None,),
              n_post: int | None = None,
              n_seeds: int = 10,
              temperature: float = 0.0,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
              answer_budget: int = DEFAULT_ANSWER_BUDGET,
              encoding_baseline: bool = False,
              verbose: bool = True) -> pd.DataFrame:
    """Full factorial sweep; n_seeds trials per cell (fresh random passphrase).

    DEFAULT = the combined A x B x H grid: naming="non-ordered",
    n_tasks in (1, 3, 7), finite_budget in (64, 256, 1024) [tokens inside
    <Thinking> only], loads (trivial, semantic_4), cot regime -> 180 trials
    at n_seeds=10. Notes:

    - ``n_tasks`` INCLUDES the passphrase task, so n_tasks=1 has no load
      tasks: the loads axis is collapsed to a single cell there and t1_load
      is logged as "none".
    - The STIMULUS is identical across finite_budgets and regimes for a given
      (n_tasks, load, seed) cell — the budget is a pure generation knob, so
      budget contrasts are paired.
    - ``finite_budgets`` may contain None (= unlimited / single pass). Any
      int budget requires the HuggingFace backend.
    - ``passphrase_last`` is a sweep AXIS since 2026-07-22: pass a bool or a
      tuple of bools (default BOTH arms: last = position-stress/anti-LITM
      headline, random = rank/lag deconfound). The two arms share every RNG
      draw up to the rank draw, so contrasts are near-paired.
    - ``anti_enumeration`` (default True) adds the I1 instruction to the cot
      template; pass False for the pre-2026-07-22 prompt. Not a sweep axis
      (run two sweeps to compare arms).
    - LEGACY runs: n_tasks_list=(None,) + finite_budgets=(None,) restores the
      old single-T1 lag design exactly (then lags/masks/n_pres/n_posts apply,
      with their old semantics and old per-cell seeds; ``n_post`` (singular)
      is still the fixed-value shorthand). These geometry axes are IGNORED
      for task-design cells.
    """
    if n_post is not None:
        n_posts = (n_post,)
    # Guard against the classic missing-comma bug: regimes=("cot") is the STRING
    # "cot", which itertools.product iterates as 'c','o','t' -> three bogus
    # regimes, none equal to "cot", so the CoT template AND the finite_budget
    # axis are silently dropped (2026-07-21 incident). Same for loads/naming.
    if isinstance(regimes, str):
        raise TypeError(
            f"regimes must be a tuple/list, got the string {regimes!r} - "
            f"did you write regimes=({regimes!r}) instead of ({regimes!r},)?")
    if isinstance(loads, str):
        raise TypeError(
            f"loads must be a tuple/list, got the string {loads!r} - "
            f"did you write loads=({loads!r}) instead of ({loads!r},)?")
    bad = [r for r in regimes if r not in ("cot", "direct")]
    if bad:
        raise ValueError(f"unknown regime(s) {bad}; expected 'cot' / 'direct'")
    # finite_budget on Ollama is supported since 2026-07-22: the native
    # /api/chat endpoint continues a prefilled assistant turn (see
    # model.OllamaBackend.continue_generate). No backend gate needed.

    if isinstance(passphrase_last, bool):
        passphrase_last = (passphrase_last,)
    pp_arms = tuple(passphrase_last)

    cells = []
    for nt in n_tasks_list:
        geoms = (list(itertools.product(lags, masks, n_pres, n_posts))
                 if nt is None else [(0, False, None, None)])
        cell_loads = tuple(loads) if (nt is None or nt > 1) else ("none",)
        # passphrase_last only changes the stimulus for task-design cells
        # (legacy path ignores it) -> collapse the axis on legacy cells.
        cell_arms = pp_arms if nt is not None else (pp_arms[0],)
        for fb, load, regime, pp, geom, seed in itertools.product(
                finite_budgets, cell_loads, regimes, cell_arms, geoms,
                range(n_seeds)):
            cells.append((nt, fb, load, regime, pp, *geom, seed))

    rows = []
    pbar = tqdm(enumerate(cells), total=len(cells), disable=not verbose)
    for k, (nt, fb, load, regime, pp, lag, mask, n_pre, n_post_, seed) in pbar:
        cfg = TrialConfig(lag=lag, t1_load=load, regime=regime, mask=mask,
                          n_pre=n_pre, n_post=n_post_,
                          seed=1000 * seed + lag + 7 * len(load)
                               + 13 * (nt or 0),
                          n_tasks=nt, naming=naming,
                          passphrase_last=pp,
                          anti_enumeration=anti_enumeration)
        rows.append(run_trial(model, tok, cfg, temperature=temperature,
                              encoding_baseline=encoding_baseline,
                              max_new_tokens=max_new_tokens,
                              finite_budget=fb,
                              answer_budget=answer_budget))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(n_tasks=nt, budget=fb, load=load)
    return pd.DataFrame(rows)
