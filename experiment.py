"""Run the lag sweep and collect both read-outs per trial.

Redesigned 2026-07-16 ("generate once, then read"): every trial runs ONE free
generation pass, and BOTH measures are read off that same trajectory:

- report_correct (bool): the generated T2 equals the correct passphrase
  -> binary "conscious" report.
- t2_total_logprob / t2_joint_prob: teacher-forced joint prob of the CORRECT
  T2 conditioned on the model's OWN generated prefix at the T2 slot (its real
  <Thinking> reasoning and its own T1 answer, cot regime) -> graded
  "unconscious strength". At temperature>0 the prefix is the actually-sampled
  one (Thomas, 2026-07-16), so the score is
  P(correct T2 | model's actually-realized reasoning).
- t1_correct (bool): sanity that the load task was actually performed.

The old empty-CoT teacher-forced score (blind to reasoning, force-fed correct
T1) is kept as an OPTIONAL control via ``encoding_baseline=True`` — it
measures "encoding strength of T2 ignoring reasoning" and lands in
``*_encoding`` columns. It is no longer the primary graded measure.
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
from .model import report_generate, sequence_logprob, DEFAULT_MAX_NEW_TOKENS

# Generation halts once the closing template tag is emitted: everything we
# need (Thinking, T1, T2) comes before it, and it keeps the large
# max_new_tokens budget cheap on trials where the model finishes early.
_STOP_AT = "</Final_Answers>"


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

# The T2 slot marker in the generated output. The scoring prefix ends right
# after this marker (plus any whitespace / opening quote the model emitted),
# i.e. exactly where the model's own T2 tokens would begin.
_T2_SLOT_RE = re.compile(r"Target 2 Result:\s*\"?")


def _t2_scoring_prefix(generated: str) -> tuple[str, bool]:
    """Split the generated text at the T2 slot for teacher-forced scoring.

    Returns (prefix, slot_missing). ``prefix`` is the model's own generated
    text up to (and including) the ``Target 2 Result:`` marker — the correct
    T2 phrase is scored conditioned on it. If the model never emitted the
    marker (or emitted it malformed), fall back to appending the marker to the
    full generated text and flag the trial (slot_missing=True) so analyses can
    gate on it, analogous to ``thinking_is_placeholder``.
    """
    m = _T2_SLOT_RE.search(generated)
    if m:
        return generated[: m.end()], False
    return generated.rstrip() + "\nTarget 2 Result: ", True


def run_trial(model, tok, cfg: TrialConfig, temperature: float = 0.0,
              encoding_baseline: bool = False,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> dict:
    tr = build_trial(cfg)

    # -- 1. generation (always runs by design) ------------------------------
    out, truncated = report_generate(
        model, tok, tr.system, tr.user,
        max_new_tokens=max_new_tokens, temperature=temperature,
        stop_at=_STOP_AT, return_truncated=True,
    )
    got_t2 = _extract_t2(out)
    got_t1 = _extract_t1(out)

    # -- 2. graded measure: P(correct T2 | model's own generated prefix) ----
    # One exact teacher-forced forward over [prompt + realized prefix + correct
    # T2]; the scoring pass itself is greedy log-softmax, only the prefix is
    # the (possibly sampled) realized one.
    prefix, slot_missing = _t2_scoring_prefix(out)
    lp = sequence_logprob(
        model, tok,
        system=tr.system,
        user_prefix=tr.user,
        target_text=tr.t2_phrase,
        prefilled_assistant=prefix,
    )

    row = {
        "lag": cfg.lag, "t1_load": cfg.t1_load, "regime": cfg.regime, "mask": cfg.mask,
        "n_pre": tr.n_pre_used, "n_post": tr.n_post_used,
        "t2_abs_index": tr.t2_abs_index,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": cfg.seed, "t2_words": cfg.t2_words,
        "t2_phrase": tr.t2_phrase,
        # graded read-out (off the model's own trajectory)
        "t2_total_logprob": lp["total_logprob"],
        "t2_mean_logprob": lp["mean_logprob"],
        "t2_joint_prob": lp["joint_prob"],
        "t2_n_tokens": lp["n_tokens"],
        # binary read-out (same generation pass)
        "report_correct": (got_t2 == tr.t2_phrase),
        "t1_correct": (got_t1 == tr.t1_answer) if tr.t1_answer else None,
        # data-quality gates
        # True = generation spent the whole max_new_tokens budget without
        # finishing (no EOS / closing tag) — the output tail was cut off.
        # Truncated trials look like report failures and their graded score
        # uses a mutilated prefix: gate analyses on this like the flags below.
        "output_truncated": truncated,
        # True = the model never emitted a usable 'Target 2 Result:' slot; the
        # graded score used the appended-marker fallback.
        "t2_slot_missing": slot_missing,
        # True = the <Thinking> block just echoed the template placeholder
        # (no real reasoning happened). cot regime only.
        "thinking_is_placeholder": (
            bool(_PLACEHOLDER_RE.search(out)) if cfg.regime == "cot" else None
        ),
        "raw_output": out,
    }

    # -- 3. optional control: old empty-CoT teacher-forced score ------------
    # P(correct T2 | stream, EMPTY reasoning, force-fed correct T1). Blind to
    # the CoT effect by construction; useful only as an encoding baseline.
    if encoding_baseline:
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


def run_sweep(model, tok, lags=(0, 1, 2, 3, 5, 8), loads=("none", "easy", "hard"),
              regimes=("cot", "direct"), masks=(False,),
              n_pres: tuple[int | None, ...] = (None,),
              n_posts: tuple[int | None, ...] = (None,),
              n_post: int | None = None,
              n_seeds: int = 20,
              temperature: float = 0.0,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
              encoding_baseline: bool = False,
              verbose: bool = True) -> pd.DataFrame:
    """Full factorial sweep. n_seeds trials per cell (each with a fresh random T2).

    Every trial generates (there is no log-prob-only mode anymore): both
    read-outs come from the same generation pass, see module docstring.

    ``n_pres`` is an independent sweep dimension (Item 2). ``None`` means
    auto-compute n_pre so the stream has exactly ``total_packets`` packets.
    Passing explicit ints lets T2 absolute position float with n_pre, so callers
    can probe the confound between n_pre and T2 position directly.

    ``n_posts`` controls the fillers after T2. ``None`` (default) draws n_post
    at random per trial in [1, budget-1] where the budget shrinks with lag/mask;
    explicit ints fix it (bounded by the same budget). ``n_post`` (singular) is
    a convenience for a single fixed value: ``n_post=3`` is shorthand for
    ``n_posts=(3,)`` and is forwarded to ``build_trial`` via ``TrialConfig``.

    ``temperature`` applies to the generation pass; at >0 the graded score is
    conditioned on the actually-sampled prefix (logged per row).

    ``max_new_tokens`` is the generation budget per trial (default 1024;
    logged per row). Generation stops early at the closing template tag, so
    the budget is only spent when the model rambles. Trials that exhaust it
    anyway are flagged ``output_truncated`` — gate analyses on that column.

    ``encoding_baseline=True`` additionally computes the legacy empty-CoT
    teacher-forced score per trial (``*_encoding`` columns) as a control.
    """
    if n_post is not None:
        n_posts = (n_post,)
    rows = []
    cells = list(itertools.product(lags, loads, regimes, masks, n_pres, n_posts,
                                   range(n_seeds)))
    pbar = tqdm(enumerate(cells), total=len(cells), disable=not verbose)
    for k, (lag, load, regime, mask, n_pre, n_post_, seed) in pbar:
        cfg = TrialConfig(lag=lag, t1_load=load, regime=regime, mask=mask,
                          n_pre=n_pre, n_post=n_post_,
                          seed=1000 * seed + lag + 7 * len(load))
        rows.append(run_trial(model, tok, cfg, temperature=temperature,
                              encoding_baseline=encoding_baseline,
                              max_new_tokens=max_new_tokens))
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(lag=lag, load=load)
    return pd.DataFrame(rows)
