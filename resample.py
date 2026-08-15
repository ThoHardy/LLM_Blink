"""Forked resampling — the on-policy graded access/report measure (issue #14, Step 1).

Each trial already produces one full trajectory. We **fork it at a fixed
structural point and draw k independent continuations at temperature 1**. The
graded measure is the fraction of continuations in which the target appears.
On-policy by construction; no teacher forcing, no normalisation set, no synthetic
slot, no candidate choice — so it runs on **both backends** and, unlike
``t2_total_logprob``, is defined on EVERY trial (reported or not).

Two fork points (§3.1, §3.2 of the ticket):

  report probe (primary): fork right after the model's own ``</Thinking>``,
      sample the ``<Final_Answers>`` block. Estimand:
      p(passphrase reaches report | this trial's realized reasoning).
  access probe:            fork right after the opening ``<Thinking>``, resample
      the whole CoT. Estimand: p(the item enters the serial stage | this stimulus).

Efficiency (§3.4): the k samples share the prompt+prefix, so on Ollama they are
dispatched concurrently across a thread pool — with OLLAMA_NUM_PARALLEL>1 the
server batches the decodes and reuses the prompt cache, so k=20 costs far less
than 20x k=1. On HF, pass num_return_sequences via the batch path.

Seeds (§3.4): every sample gets a DISTINCT seed drawn from a Generator seeded by
``seed`` — a SEPARATE stream from the stimulus RNG (which ``rescore_graded``
replays), so saved CSVs stay bit-exact. With a fixed seed Ollama is deterministic
and returns k identical strings; ``fork_samples`` asserts the samples are not all
identical (the single most likely silent failure in this design).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .model import continue_generate, OllamaBackend
from .protocol import STOP_THINKING, STOP_ANSWERS
from .readout import parse_task_report_names, answer_contains, t2_in_cot


# The CoT ends at the answer-block opener <Final_Answers> (reliable, ~99%), not
# at </Thinking> (which small models often omit) — see Trajectory.offset. So the
# access probe (resample the CoT) stops there; the report probe stops at the
# answer-block CLOSER.
_ANSWERS_OPEN = "<Final_Answers>"
_DEFAULT_STOP = {"pre_cot": _ANSWERS_OPEN, "post_cot": STOP_ANSWERS,
                 "response": STOP_ANSWERS}
_DEFAULT_BUDGET = {"pre_cot": 640, "post_cot": 256, "response": 256}


class DegenerateForkError(RuntimeError):
    """All k forked samples were byte-identical (temperature/seed not effective).

    Carries the offending ``samples`` so the caller can log ``probe_degenerate``
    and inspect them rather than losing the run.
    """
    def __init__(self, samples):
        self.samples = samples
        super().__init__(
            f"all {len(samples)} forked samples are byte-identical - is "
            "temperature>0 and a DISTINCT seed passed per sample? (Ollama with a "
            "fixed seed is deterministic; see #14 §3.4)")


def sample_seeds(seed: int, k: int):
    """k distinct sampling seeds from a Generator SEPARATE from the stimulus RNG."""
    return [int(x) for x in
            np.random.default_rng(seed).integers(1, 2**31 - 1, size=k)]


def fork_samples(model, tok, trial, traj, at: str = "post_cot", k: int = 20,
                 temperature: float = 1.0, seed: int = 0, stop: str | None = None,
                 max_new_tokens: int | None = None, n_workers: int = 8,
                 raise_on_degenerate: bool = True) -> list[str]:
    """Draw k continuations from ``prompt + traj.text[:fork_offset]``.

    Returns the k raw continuation strings (the prefix is NOT included; parsing
    is the caller's job). Raises :class:`DegenerateForkError` if all k are
    byte-identical (unless ``raise_on_degenerate=False``).
    """
    off = traj.offset(at)
    if off is None:
        raise ValueError(
            f"fork point {at!r} is not present in this trajectory "
            f"(regime without a <Thinking> block, or missing close tag)")
    prefix = traj.text[:off]
    if stop is None:
        stop = _DEFAULT_STOP.get(at)
    if max_new_tokens is None:
        max_new_tokens = _DEFAULT_BUDGET.get(at, 256)
    seeds = sample_seeds(seed, k)

    def _one(sd):
        text, _n, _trunc = continue_generate(
            model, tok, trial.system, trial.user,
            prefilled_assistant=prefix, max_new_tokens=max_new_tokens,
            temperature=temperature, stop_at=stop, seed=sd)
        return text

    if isinstance(model, OllamaBackend) and n_workers > 1 and k > 1:
        with ThreadPoolExecutor(max_workers=min(n_workers, k)) as ex:
            samples = list(ex.map(_one, seeds))
    else:
        # HF path (and n_workers==1): sequential. A batched num_return_sequences
        # path lives in model._continue_generate_hf_batch when torch is present.
        samples = [_one(sd) for sd in seeds]

    if raise_on_degenerate and k > 1 and len(set(samples)) == 1:
        raise DegenerateForkError(samples)
    return samples


# ---------------------------------------------------------------------------
# per-sample scoring -> counts (s, k) for the beta-binomial mixture
# ---------------------------------------------------------------------------

def score_report_samples(samples, t2_task_name: str, t2_phrase: str) -> dict:
    """report_rate: fraction of sampled reports whose passphrase line contains
    the passphrase (word-bounded ``report_contains`` logic, applied per sample).

    Returns dict(s, k, degenerate, samples_ok). Each sample is parsed with the
    existing tolerant free-report parser; a hit = the passphrase task's reported
    answer contains the passphrase (echo-text tolerant, project's access defn).
    """
    hits = 0
    for text in samples:
        parsed = parse_task_report_names(text, [t2_task_name])
        resp = parsed["reported"].get(str(t2_task_name).upper())
        if answer_contains(resp, t2_phrase):
            hits += 1
    return dict(s=int(hits), k=int(len(samples)),
                degenerate=(len(set(samples)) == 1 and len(samples) > 1))


def score_access_samples(samples, t2_phrase: str,
                         t2_task_name: str | None = None) -> dict:
    """access_rate: fraction of sampled CoTs in which the passphrase task is
    mentioned or processed (by name or content) — the ``t2_in_cot`` logic."""
    hits = sum(1 for text in samples
               if t2_in_cot(text, t2_phrase, t2_task_name))
    return dict(s=int(hits), k=int(len(samples)),
                degenerate=(len(set(samples)) == 1 and len(samples) > 1))


# ---------------------------------------------------------------------------
# nested probe (§3.3 / §4.3): sample CoTs, then reports within each CoT
# ---------------------------------------------------------------------------

def nested_fork(model, tok, trial, traj, k_cot: int = 8, k_rep: int = 8,
                temperature: float = 1.0, seed: int = 0, n_workers: int = 8):
    """For each of k_cot sampled CoTs, draw k_rep reports from its transition.

    The variance decomposition teacher-forcing cannot produce (§4.3): if report
    is near-deterministic GIVEN a CoT (within-CoT variance ~0, all variance
    BETWEEN CoTs) access is decided in the workspace = **ignition**; substantial
    within-CoT variance = the read-out is itself stochastic = **graded**.

    Returns list of dicts, one per sampled CoT:
      {cot_id, s (reports containing the passphrase), k (=k_rep),
       t2_in_this_cot (was the passphrase task in the sampled CoT)}.
    Feed the (cot_id, per-report 0/1) to ``mixture.icc_nested``.
    """
    pre_off = traj.offset("pre_cot")
    if pre_off is None:
        raise ValueError("no <Thinking> opener to fork the CoT from")
    cot_prefix = traj.text[:pre_off]                      # prompt + '<Thinking>\n'
    cots = fork_samples(model, tok, trial, traj, at="pre_cot", k=k_cot,
                        temperature=temperature, seed=seed, n_workers=n_workers,
                        raise_on_degenerate=False)
    out = []
    for j, cot in enumerate(cots):
        # rebuild a trajectory whose reasoning IS this sampled CoT, then fork it.
        text = cot_prefix + cot.rstrip() + "\n" + _ANSWERS_OPEN + "\n"
        pseudo = _PseudoTraj(text)
        reps = fork_samples(model, tok, trial, pseudo, at="post_cot", k=k_rep,
                            temperature=temperature, seed=seed * 131 + j + 1,
                            n_workers=n_workers, raise_on_degenerate=False)
        sc = score_report_samples(reps, trial.t2_task_name, trial.t2_phrase)
        out.append(dict(cot_id=j, s=sc["s"], k=sc["k"],
                        t2_in_this_cot=int(t2_in_cot(cot, trial.t2_phrase,
                                                     trial.t2_task_name))))
    return out


class _PseudoTraj:
    """Minimal Trajectory-like object exposing .text and .offset for a rebuilt
    prompt+CoT+transition string (used by the nested probe)."""
    _ANSWERS_OPEN = "<Final_Answers>"   # Trajectory.offset reads self._ANSWERS_OPEN

    def __init__(self, text):
        self.text = text

    def offset(self, at):
        from .protocol import Trajectory
        return Trajectory.offset(self, at)
