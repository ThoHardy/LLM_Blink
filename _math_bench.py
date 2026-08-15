"""MATH-Algebra load bank (issue #14 Step 6).

Loads the static JSON bank built by `build_math_bench.py` from the Hendrycks
et al. (2021) MATH dataset (Algebra subset, `EleutherAI/hendrycks_math` mirror),
filtered to compact problems with short unambiguous INTEGER answers. Exposes it
in the same interface as `stimuli.T1_SEMANTIC_BANKS`: a dict keyed by MATH
difficulty Level 1..5, each a tuple of `(question, answer)` items.

Using the standard benchmark (rather than our programmatic `T1_MATH_BANKS`)
answers the "your difficulty levels are self-defined" objection: the levels are
the dataset authors' own. No network or `datasets` library at run time — the
bank is a committed JSON file.
"""
from __future__ import annotations
import json
import os

_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", "math_bench_algebra.json")


def _load():
    if not os.path.exists(_JSON):
        raise FileNotFoundError(
            f"MATH bank {_JSON} missing — run `python3 build_math_bench.py` "
            "once (needs network) to fetch and filter the Hendrycks MATH "
            "Algebra subset.")
    with open(_JSON) as f:
        raw = json.load(f)
    # keys "1".."5" -> int level; items list[list] -> tuple[tuple]
    return {int(k): tuple((q, a) for q, a in v) for k, v in raw.items()}


# {1: ((q, a), ...), ..., 5: (...)} — MATH difficulty Level 1..5.
MATH_BENCH_BANKS = _load()
