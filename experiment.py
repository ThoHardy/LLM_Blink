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


# Text the cot OUTPUT TEMPLATE uses as its <Thinking> placeholder. If it shows up
# verbatim in the output, the model copied the template instead of reasoning.
_PLACEHOLDER_RE = re.compile(r"\[process the stream")


def run_trial(model, tok, cfg: TrialConfig, do_generation: bool = True,
              temperature: float = 0.0) -> dict:
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
        "n_pre": tr.n_pre_used, "n_post": tr.n_post_used,
        "t2_abs_index": tr.t2_abs_index,
        "temperature": temperature,
        "seed": cfg.seed, "t2_words": cfg.t2_words,
        "t2_phrase": tr.t2_phrase,
        "t2_total_logprob": lp["total_logprob"],
        "t2_mean_logprob": lp["mean_logprob"],
        "t2_joint_prob": lp["joint_prob"],
        "t2_n_tokens": lp["n_tokens"],
    }

    if do_generation:
        out = report_generate(model, tok, tr.system, tr.user_full,
                              temperature=temperature)
        got_t2 = _extract_t2(out)
        got_t1 = _extract_t1(out)
        row["report_correct"] = (got_t2 == tr.t2_phrase)
        row["t1_correct"] = (got_t1 == tr.t1_answer) if tr.t1_answer else None
        # Data-quality gate for the cot regime: True means the <Thinking> block
        # just echoed the template placeholder (no real reasoning happened).
        row["thinking_is_placeholder"] = (
            bool(_PLACEHOLDER_RE.search(out)) if cfg.regime == "cot" else None
        )
        row["raw_output"] = out
    return row


def run_sweep(model, tok, lags=(0, 1, 2, 3, 5, 8), loads=("none", "easy", "hard"),
              regimes=("cot", "direct"), masks=(False,),
              n_pres: tuple[int | None, ...] = (None,),
              n_posts: tuple[int | None, ...] = (None,),
              n_seeds: int = 20,
              temperature: float = 0.0,
              do_generation: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Full factorial sweep. n_seeds trials per cell (each with a fresh random T2).

    ``n_pres`` is an independent sweep dimension (Item 2). ``None`` means
    auto-compute n_pre so the stream has exactly ``total_packets`` packets.
    Passing explicit ints lets T2 absolute position float with n_pre, so callers
    can probe the confound between n_pre and T2 position directly.

    ``n_posts`` controls the fillers after T2. ``None`` (default) draws n_post
    at random per trial in [0, budget] where the budget shrinks with lag/mask;
    explicit ints fix it (bounded by the same budget).

    ``temperature`` is passed to the generation scorer (0.0 = greedy, the
    default). It only affects the binary report measure, never log-prob scoring.
    Sensible non-zero values: 0.3, 0.7, 1.0.
    """
    rows = []
    cells = list(itertools.product(lags, loads, regimes, masks, n_pres, n_posts,
                                   range(n_seeds)))
    pbar = tqdm(enumerate(cells), total=len(cells), disable=not verbose)
    for k, (lag, load, regime, mask, n_pre, n_post, seed) in pbar:
        cfg = TrialConfig(lag=lag, t1_load=load, regime=regime, mask=mask,
                          n_pre=n_pre, n_post=n_post,
                          seed=1000 * seed + lag + 7 * len(load))
        rows.append(run_trial(model, tok, cfg, do_generation=do_generation,
                              temperature=temperature))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(lag=lag, load=load)
    return pd.DataFrame(rows)
