"""Run the lag sweep and collect both read-outs per trial.

Each trial yields:
- report_correct (bool): T2 appears correctly in greedy generation  -> "conscious"
- t2_total_logprob / t2_joint_prob: teacher-forced joint prob of correct T2 -> "unconscious"
- t1_correct (bool): sanity that the load task was actually performed
"""
from __future__ import annotations
import re
import itertools
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # keep package dependency-light
    def tqdm(iterable, **kwargs):
        return iterable

from .stimuli import TrialConfig, build_trial
from .model import report_generate, sequence_logprob


def _extract_t2(text: str) -> str | None:
    m = re.search(r"Target 2 Result:\s*\"?([A-Z ]+?)\"?\s*(?:\n|$)", text)
    return m.group(1).strip() if m else None


def _extract_t1(text: str) -> str | None:
    # answers may be numbers (math loads) or short uppercase words (semantic loads)
    m = re.search(r"Target 1 Result:\s*\"?([^\"\n]+?)\"?\s*(?:\n|$)", text)
    return m.group(1).strip().upper() if m else None


def run_trial(model, tok, cfg: TrialConfig, do_generation: bool = True) -> dict:
    tr = build_trial(cfg)

    # graded "unconscious" measure: joint log-prob of the correct T2 under the answer template
    lp = sequence_logprob(
        model, tok,
        system=tr.system,
        user_prefix=tr.user_prefix,
        target_text=tr.t2_phrase,
        prefilled_assistant=tr.template_prefix_after_prompt,
    )

    row = {
        "lag": cfg.lag, "t1_load": cfg.t1_load, "regime": cfg.regime, "mask": cfg.mask,
        "seed": cfg.seed, "t2_words": cfg.t2_words,
        "t2_phrase": tr.t2_phrase,
        "t2_total_logprob": lp["total_logprob"],
        "t2_mean_logprob": lp["mean_logprob"],
        "t2_joint_prob": lp["joint_prob"],
        "t2_n_tokens": lp["n_tokens"],
    }

    if do_generation:
        out = report_generate(model, tok, tr.system, tr.user_full)
        got_t2 = _extract_t2(out)
        got_t1 = _extract_t1(out)
        row["report_correct"] = (got_t2 == tr.t2_phrase)
        row["t1_correct"] = (got_t1 == tr.t1_answer) if tr.t1_answer else None
        row["raw_output"] = out
    return row


def run_sweep(model, tok, lags=(0, 1, 2, 3, 5, 8), loads=("none", "easy", "hard"),
              regimes=("cot", "direct"), masks=(False,), n_seeds: int = 20,
              do_generation: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Full factorial sweep. n_seeds trials per cell (each with a fresh random T2)."""
    rows = []
    cells = list(itertools.product(lags, loads, regimes, masks, range(n_seeds)))
    pbar = tqdm(enumerate(cells), total=len(cells), disable=not verbose)
    for k, (lag, load, regime, mask, seed) in pbar:
        cfg = TrialConfig(lag=lag, t1_load=load, regime=regime, mask=mask,
                          seed=1000 * seed + lag + 7 * len(load))
        rows.append(run_trial(model, tok, cfg, do_generation=do_generation))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(lag=lag, load=load)
    return pd.DataFrame(rows)
