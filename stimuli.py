"""Build RSVP-style packet streams with controllable T1 load, T2 content, and lag.

Key design choices (see ../PROMPTS.md and ../LITERATURE.md §C):
- T2 is a NOVEL random NATO-word triplet each trial -> joint-P(T2) is not at ceiling.
- T2 absolute position is held ~constant by padding before T1 (controls 'lost-in-the-middle').
- T1 has five load levels per track: semantic_0..4 and math_0..4 (plus 'none' baseline).
  Aliases: easy=semantic_0, hard=semantic_1, easy_math=math_0, hard_math=math_1.
- Optional post-T1 'mask' packet (the human AB needs a mask).
- Two output regimes: 'direct' (answer T2 first, no scratchpad) and 'cot' (solve T1 first).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

NATO = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL",
        "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA",
        "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM", "VICTOR", "WHISKEY",
        "XRAY", "YANKEE", "ZULU"]

FILLERS = [
    "System initialization complete. Weather is clear.",
    "Update {v} applied to main server.",
    "Security camera {n} offline.",
    "Maintenance scheduled for Tuesday.",
    "CPU temperature nominal at {n}C.",
    "Disk usage at {n} percent.",
    "Network latency {n} ms.",
    "Backup job {n} finished successfully.",
    "Sensor array recalibrated.",
    "Login from internal node {n}.",
]

# ---------------------------------------------------------------------------
# T1 SEMANTIC BANKS — five levels of inference load (see PROMPTS.md §P1).
# Each item: (instruction, expected_answer).  Answers are single uppercase
# words so _extract_t1 in experiment.py can score them uniformly.
#
# Level 0 — one-step categorisation (direct feature lookup)
# Level 1 — two-premise syllogism / transitive chain (novel words, classic traps)
# Level 2 — three-step chain or two-premise negation (Celarent / Darii / Ferio)
# Level 3 — four-step transitive or three-premise chain + distractor premises
# Level 4 — five premises; 1-2 distractor premises designed to elicit errors
# ---------------------------------------------------------------------------

T1_SEMANTIC_EASY = [   # level 0
    ("Decide whether DOLPHIN is an ANIMAL or a TOOL. Answer ANIMAL or TOOL.", "ANIMAL"),
    ("Decide whether HAMMER is an ANIMAL or a TOOL. Answer ANIMAL or TOOL.", "TOOL"),
    ("Decide whether COPPER is a METAL or a FRUIT. Answer METAL or FRUIT.", "METAL"),
    ("Decide whether MANGO is a METAL or a FRUIT. Answer METAL or FRUIT.", "FRUIT"),
    ("Decide whether VIOLIN is a VEHICLE or an INSTRUMENT. Answer VEHICLE or INSTRUMENT.",
     "INSTRUMENT"),
]

T1_SEMANTIC_HARD = [   # level 1
    ("All gleeps are morks. All morks are florks. Does it follow that all gleeps are florks? "
     "Answer VALID or INVALID.", "VALID"),
    ("All gleeps are morks. Some morks are florks. Does it follow that some gleeps are florks? "
     "Answer VALID or INVALID.", "INVALID"),
    ("Box A is heavier than Box B. Box B is heavier than Box C. Is Box A heavier than Box C? "
     "Answer YES or NO.", "YES"),
    ("Tom finished before Sara. Sara finished before Leo. Did Leo finish before Tom? "
     "Answer YES or NO.", "NO"),
    ("No wugs are zors. All zors are blims. Does it follow that no wugs are blims? "
     "Answer VALID or INVALID.", "INVALID"),
]

T1_SEMANTIC_L2 = [   # level 2 — three-step chain / two-premise negation
    # Celarent: All M are G, No G are S -> No M are S
    ("All morks are gleeps. No gleeps are snorfs. "
     "Does it follow that no morks are snorfs? Answer VALID or INVALID.", "VALID"),
    # Darii: Some V are B, All B are T -> Some V are T
    ("Some vorps are blims. All blims are trens. "
     "Does it follow that some vorps are trens? Answer VALID or INVALID.", "VALID"),
    # Ferio trap: All F are W, Some W are not G ≠ Some F are not G
    ("All flurbs are wumps. Some wumps are not gleeks. "
     "Does it follow that some flurbs are not gleeks? Answer VALID or INVALID.", "INVALID"),
    # Three-step transitive chain (non-adjacent pair)
    ("Alice is older than Bob. Bob is older than Carol. Carol is older than Dave. "
     "Is Alice older than Dave? Answer YES or NO.", "YES"),
    # Three-step ordering; question targets the boundary
    ("Pax is heavier than Quin. Quin is heavier than Remy. Remy is heavier than Sven. "
     "Is Sven the lightest of the four? Answer YES or NO.", "YES"),
]

T1_SEMANTIC_L3 = [   # level 3 — four-step chain or three-premise + distractor
    # Four-step transitive; question is inverted (bottom vs top)
    ("A outranks B. C outranks A. D outranks C. E outranks D. "
     "Does B outrank E? Answer YES or NO.", "NO"),
    # All->All->No chain (must traverse three premises)
    ("All vorps are blims. All blims are snorfs. No snorfs are gleeks. "
     "Does it follow that no vorps are gleeks? Answer VALID or INVALID.", "VALID"),
    # Some->All->No: some W are Z, Z->M, No M are T => some W are not T
    ("Some wugs are zorps. Every zorp is a mib. No mib is a trel. "
     "Does it follow that some wugs are not trils? Answer VALID or INVALID.", "VALID"),
    # Invalid: All G are F, Some F are S, All S are W; G might be non-S flurbs
    ("All gleeks are flurbs. Some flurbs are snorfs. All snorfs are wumps. "
     "Does it follow that some gleeks are wumps? Answer VALID or INVALID.", "INVALID"),
    # Every M is V, Some V are B, No B is M => some V are not M
    ("Every mork is a vorp. Some vorps are blims. No blim is a mork. "
     "Does it follow that some vorps are not morks? Answer VALID or INVALID.", "VALID"),
]

T1_SEMANTIC_L4 = [   # level 4 — five premises, distractor traps
    # Z->M->S, No S->F | last two premises (F->W, some W are Z) are traps
    ("All zorps are morks. Every mork is a snick. No snick is a flurb. "
     "Every flurb is a wump. Some wumps are zorps. "
     "Does it follow that no zorps are flurbs? Answer VALID or INVALID.", "VALID"),
    # No W are G => No G are W => All G not-W; other premises are distractors
    ("Some gleeks are blims. All blims are snorfs. Some snorfs are vorps. "
     "All vorps are wumps. No wumps are gleeks. "
     "Does it follow that some gleeks are not wumps? Answer VALID or INVALID.", "VALID"),
    # Five-step transitive; question on non-adjacent pair
    ("A outranks B. B outranks C. C outranks D. D outranks E. E outranks F. "
     "Does C outrank F? Answer YES or NO.", "YES"),
    # All V->B, All S->G, some B are S, No G are W; NOT all V are W (distractors mislead)
    ("All vorps are blims. All snorfs are gleeks. Some blims are snorfs. "
     "No gleeks are wumps. All wumps are trens. "
     "Does it follow that no vorps are wumps? Answer VALID or INVALID.", "INVALID"),
    # Ordering puzzle with five constraints; unique solution
    # R<T<P<Q<S; question: who is third?
    ("There are five runners: P, Q, R, S, T. "
     "P finished before Q. R finished before P. S finished after Q. "
     "T finished between R and P (i.e., after R but before P). "
     "Who finished in third place? Answer P, Q, R, S, or T.", "P"),
]

# Convenience lookup used by _get_t1
T1_SEMANTIC_BANKS = {
    0: T1_SEMANTIC_EASY,
    1: T1_SEMANTIC_HARD,
    2: T1_SEMANTIC_L2,
    3: T1_SEMANTIC_L3,
    4: T1_SEMANTIC_L4,
}

# ---------------------------------------------------------------------------
# T1 MATH BANKS — five levels (see PROMPTS.md §P1).
# All answers are integers as strings so scoring stays uniform.
#
# Level 0 — single arithmetic operation
# Level 1 — enumerate primes in a range, sum, multiply
# Level 2 — 3–4 steps: enumerate small set + combine
# Level 3 — 4–5 steps: two enumerations or modular/divisibility
# Level 4 — 5+ steps: nested enumerations or constraint-based search
# ---------------------------------------------------------------------------

T1_MATH_BANKS = {
    0: [   # single operation
        ("Compute 2 + 3.", "5"),
        ("Compute 9 - 4.", "5"),
        ("Compute 3 * 4.", "12"),
        ("Compute 15 - 7.", "8"),
        ("Compute 6 + 8.", "14"),
    ],
    1: [   # enumerate primes / small combinatorics
        # primes in (50,70): 53,59,61,67 -> sum 240 -> *3 = 720
        ("Calculate the sum of all prime numbers between 50 and 70, "
         "then multiply by 3.", "720"),
        # primes in (60,80): 61,67,71,73,79 -> sum 351 -> *2 = 702
        ("Calculate the sum of all prime numbers between 60 and 80, "
         "then multiply by 2.", "702"),
        # single-digit primes: 2*3*5*7 = 210
        ("Multiply together all single-digit prime numbers.", "210"),
        # perfect squares 1-50: 1,4,9,16,25,36,49 -> sum 140
        ("Sum all perfect squares from 1 to 50 inclusive.", "140"),
        # odd numbers 11-29: sum 200, minus 50 = 150
        ("Sum all odd numbers from 11 to 29 inclusive, then subtract 50.", "150"),
    ],
    2: [   # 3-4 steps
        # primes 20-30: 23, 29 -> 23*29 = 667
        ("Find all prime numbers between 20 and 30. "
         "Multiply them together.", "667"),
        # 3! + 4! = 6 + 24 = 30
        ("Compute 3 factorial plus 4 factorial.", "30"),
        # even 2..16: 2+4+6+8+10+12+14+16 = 72; 72/4 = 18
        ("Sum all even numbers from 2 to 16 inclusive, then divide by 4.", "18"),
        # 12^2 = 144; primes < 20: 2,3,5,7,11,13,17,19 = 8; 144 - 8 = 136
        ("What is 12 squared minus the count of prime numbers less than 20?", "136"),
        # factors of 48: 1,2,3,4,6,8,12,16,24,48 = 10; 10*7 = 70
        ("List all factors of 48. How many are there? Multiply that count by 7.", "70"),
    ],
    3: [   # 4-5 steps
        # primes 1-30: 2,3,5,7,11,13,17,19,23,29 -> sum 129; 129 mod 11 = 8
        ("Sum all prime numbers from 1 to 30 inclusive. "
         "What is the remainder when that sum is divided by 11?", "8"),
        # two-digit perfect squares: 16,25,36,49,64,81 -> sum 271; 5^3=125; 271-125=146
        ("Compute the sum of all two-digit perfect squares, "
         "then subtract 5 cubed.", "146"),
        # 100 minus each prime < 20: 100-2-3-5-7-11-13-17-19 = 23
        ("Start with 100. Subtract every prime number less than 20 in increasing order. "
         "What is the result?", "23"),
        # 1-50 divisible by 3 or 7 but not both: 14+5 = 19
        ("How many integers from 1 to 50 are divisible by either 3 or 7, "
         "but not by both 3 and 7?", "19"),
        # largest two-digit prime ≡ 1 (mod 6): 97
        ("What is the largest two-digit prime number that is also "
         "one more than a multiple of 6?", "97"),
    ],
    4: [   # 5+ steps / nested enumeration
        # primes < 20: squares sum = 4+9+25+49+121+169+289+361 = 1027
        ("Compute the sum of the squares of all prime numbers less than 20.", "1027"),
        # odd multiple of 7, 100-200, digit sum = 11: only 119
        ("N is an odd multiple of 7 strictly between 100 and 200. "
         "The sum of N's digits equals 11. What is N?", "119"),
        # div by 2 not 3 not 5 in 1-100: 27; *3 = 81
        ("How many integers from 1 to 100 are divisible by 2 "
         "but not divisible by 3 and not divisible by 5? "
         "Multiply that count by 3.", "81"),
        # double-and-add-1 five times from 2: 2->5->11->23->47->95
        ("Start with 2. Repeat five times: double the current number and add 1. "
         "What is the final result?", "95"),
        # two-digit perfect squares: 16..81 (6 of them); 81*16=1296; 1296+6=1302
        ("Find all two-digit perfect squares. Multiply the largest by the smallest, "
         "then add the total count of such numbers.", "1302"),
    ],
}

# ---------------------------------------------------------------------------
# Pool expansion: keep the 5 hand-curated heads frozen, then extend each pool
# to 100 items using deterministic, seeded template generators (see
# _t1_generators.py). All math answers are computed programmatically.
# ---------------------------------------------------------------------------

from . import _t1_generators as _gen


def _extend_to_100(pool: list, generator, *, target: int = 100, seed: int,
                   max_attempts: int = 5) -> None:
    """Extend `pool` with items from `generator(n, seed)` until len(pool) == target.

    Generated questions that collide with already-present questions are dropped.
    If the first call doesn't supply enough unique items, retry with bumped seeds.
    """
    have = {q for q, _ in pool}
    attempt = 0
    while len(pool) < target and attempt < max_attempts:
        needed = target - len(pool)
        # First attempt: ask for exactly `needed` so the generator preserves
        # the bucket balance designed in (5,4,3-template etc).
        # On retry, ask for extras to overcome the few collisions.
        ask = needed if attempt == 0 else needed + 8
        candidates = generator(n=ask, seed=seed + attempt * 1000)
        for q, a in candidates:
            if q in have:
                continue
            have.add(q)
            pool.append((q, a))
            if len(pool) >= target:
                break
        attempt += 1
    if len(pool) < target:
        raise RuntimeError(
            f"Could not fill pool to {target} items (got {len(pool)}); "
            f"generator may need a wider template space."
        )


_extend_to_100(T1_MATH_BANKS[0], _gen.gen_math_l0, seed=100)
_extend_to_100(T1_MATH_BANKS[1], _gen.gen_math_l1, seed=101)
_extend_to_100(T1_MATH_BANKS[2], _gen.gen_math_l2, seed=102)
_extend_to_100(T1_MATH_BANKS[3], _gen.gen_math_l3, seed=103)
_extend_to_100(T1_MATH_BANKS[4], _gen.gen_math_l4, seed=104)

_extend_to_100(T1_SEMANTIC_BANKS[0], _gen.gen_semantic_l0, seed=200)
_extend_to_100(T1_SEMANTIC_BANKS[1], _gen.gen_semantic_l1, seed=201)
_extend_to_100(T1_SEMANTIC_BANKS[2], _gen.gen_semantic_l2, seed=202)
_extend_to_100(T1_SEMANTIC_BANKS[3], _gen.gen_semantic_l3, seed=203)
_extend_to_100(T1_SEMANTIC_BANKS[4], _gen.gen_semantic_l4, seed=204)

# Legacy dict kept for backward compatibility
T1_MATH = {
    "easy_math": T1_MATH_BANKS[0][0],
    "hard_math": T1_MATH_BANKS[1][0],
}

# ---------------------------------------------------------------------------
# Load dispatcher
# ---------------------------------------------------------------------------

_SEMANTIC_ALIASES = {
    "easy": 0, "semantic_0": 0,
    "hard": 1, "semantic_1": 1,
    "semantic_2": 2,
    "semantic_3": 3,
    "semantic_4": 4,
}
_MATH_ALIASES = {
    "easy_math": 0, "math_0": 0,
    "hard_math": 1, "math_1": 1,
    "math_2": 2,
    "math_3": 3,
    "math_4": 4,
}
_VALID_LOADS = (
    "none",
    *_SEMANTIC_ALIASES,
    *_MATH_ALIASES,
)


def _get_t1(load: str, rng: random.Random):
    """Returns (t1_line, t1_answer, tag).  load=='none' -> (None, None, '')."""
    if load == "none":
        return None, None, ""
    if load in _SEMANTIC_ALIASES:
        task, ans = rng.choice(T1_SEMANTIC_BANKS[_SEMANTIC_ALIASES[load]])
    elif load in _MATH_ALIASES:
        task, ans = rng.choice(T1_MATH_BANKS[_MATH_ALIASES[load]])
    else:
        raise ValueError(
            f"Unknown t1_load: {load!r}. "
            f"Valid values: {', '.join(_VALID_LOADS)}"
        )
    return f"LOGIC OVERRIDE. {task}", ans, " - T1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filler(rng: random.Random) -> str:
    return rng.choice(FILLERS).format(v=f"1.0.{rng.randint(0,9)}", n=rng.randint(1, 99))


def random_passphrase(rng: random.Random, n_words: int = 3) -> str:
    return " ".join(rng.sample(NATO, n_words))


# ---------------------------------------------------------------------------
# Trial dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    lag: int = 2                  # packets strictly between T1 and T2
    t1_load: str = "hard"         # see _VALID_LOADS
    regime: str = "cot"           # direct | cot
    mask: bool = False            # post-T1 mask packet
    n_pre: int | None = None      # fillers before T1 (auto: keeps T2 pos ~constant if None)
    n_post: int = 2               # fillers after T2 before End of stream
    t2_words: int = 3
    target_t2_abs_index: int = 12  # desired absolute packet index of T2 (position control)
    seed: int = 0


@dataclass
class Trial:
    system: str
    user_prefix: str             # everything up to (excluding) the T2 passphrase string
    user_full: str               # full prompt incl. T2 + template (for generation)
    t2_phrase: str
    t1_answer: str | None
    template_prefix_after_prompt: str  # assistant text preceding the T2 slot, for logprob scoring
    config: TrialConfig = field(default=None)


SYSTEM = "You are an automated data-extraction system."

RULES = ("You will be provided with a stream of sequential data packets. Read the entire stream, "
         "then output EXACTLY the template at the end. Do not add conversational text, "
         "introductory phrases, or extra punctuation.")


def build_trial(cfg: TrialConfig) -> Trial:
    rng = random.Random(cfg.seed)
    t2 = random_passphrase(rng, cfg.t2_words)

    t1_line, t1_ans, t1_tag = _get_t1(cfg.t1_load, rng)
    if t1_line is None:                  # 'none' load -> neutral filler in the T1 slot
        t1_line = _filler(rng)

    # position control: pad before T1 so that T2 lands at target_t2_abs_index
    # layout: [n_pre fillers][T1][mask?][lag fillers][T2]
    pre_blocks = 1 + (1 if cfg.mask else 0) + cfg.lag
    if cfg.n_pre is None:
        n_pre = max(0, cfg.target_t2_abs_index - 1 - pre_blocks)
    else:
        n_pre = cfg.n_pre

    packets = []
    for _ in range(n_pre):
        packets.append(_filler(rng))
    packets.append((t1_line, t1_tag))
    if cfg.mask:
        packets.append(f"Reference codes logged: {random_passphrase(rng, 3)}.")
    for _ in range(cfg.lag):
        packets.append(_filler(rng))
    packets.append((f'The secure operational passphrase is "{t2}".', " - T2"))
    for _ in range(cfg.n_post):
        packets.append(_filler(rng))
    packets.append("End of stream.")

    lines = []
    for i, p in enumerate(packets, start=1):
        if isinstance(p, tuple):
            text, tag = p
            lines.append(f"[Packet {i:02d}{tag}]: {text}")
        else:
            lines.append(f"[Packet {i:02d}]: {p}")
    stream = "\n".join(lines)

    if cfg.regime == "direct":
        template = ('OUTPUT TEMPLATE:\n<Final_Answers>\n'
                    'Target 2 Result: [the exact 3-word passphrase]\n'
                    'Target 1 Result: [the exact final number]\n</Final_Answers>')
        assistant_prefix = "<Final_Answers>\nTarget 2 Result: "
    else:  # cot
        template = ('OUTPUT TEMPLATE:\n<Thinking>\n[process the stream, do the math]\n</Thinking>\n'
                    '<Final_Answers>\nTarget 1 Result: [the exact final number]\n'
                    'Target 2 Result: [the exact 3-word passphrase]\n</Final_Answers>')
        t1a = t1_ans if t1_ans is not None else "N/A"
        assistant_prefix = (f"<Thinking>\n</Thinking>\n<Final_Answers>\n"
                            f"Target 1 Result: {t1a}\nTarget 2 Result: ")

    header = f"{RULES}\nDATA STREAM:\n{stream}\n{template}"
    return Trial(
        system=SYSTEM,
        user_prefix=header,
        user_full=header,
        t2_phrase=t2,
        t1_answer=t1_ans,
        template_prefix_after_prompt=assistant_prefix,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Optional validation: run only when LLM_BLINK_VALIDATE=1 or when this module
# is executed as a script. Never runs on a normal import.
# ---------------------------------------------------------------------------

def _validate_t1_pools(verbose: bool = False) -> None:
    """Assert pool size / type / balance / regeneration invariants.

    Checks:
      1. Each of the 10 pools has exactly 100 items.
      2. No duplicate instruction strings within a pool.
      3. All items are (str, str) tuples.
      4. Semantic pools: VALID/INVALID and YES/NO label balance within +/-10%
         (relative to the smaller count). L0's category-name answers are
         exempt from this since their answer space is intrinsically wide.
      5. Math pools: re-run each generator with the same seed and assert the
         tail (items past index 5) matches exactly. This ensures answers are
         deterministically derivable and not drifted.
    """
    from collections import Counter
    from . import _t1_generators as _gen_check

    expected_size = 100
    head_size = 5

    pool_specs = [
        ("semantic_0", T1_SEMANTIC_BANKS[0], _gen_check.gen_semantic_l0, 200),
        ("semantic_1", T1_SEMANTIC_BANKS[1], _gen_check.gen_semantic_l1, 201),
        ("semantic_2", T1_SEMANTIC_BANKS[2], _gen_check.gen_semantic_l2, 202),
        ("semantic_3", T1_SEMANTIC_BANKS[3], _gen_check.gen_semantic_l3, 203),
        ("semantic_4", T1_SEMANTIC_BANKS[4], _gen_check.gen_semantic_l4, 204),
        ("math_0", T1_MATH_BANKS[0], _gen_check.gen_math_l0, 100),
        ("math_1", T1_MATH_BANKS[1], _gen_check.gen_math_l1, 101),
        ("math_2", T1_MATH_BANKS[2], _gen_check.gen_math_l2, 102),
        ("math_3", T1_MATH_BANKS[3], _gen_check.gen_math_l3, 103),
        ("math_4", T1_MATH_BANKS[4], _gen_check.gen_math_l4, 104),
    ]

    for name, pool, gen_fn, seed in pool_specs:
        # (1) size
        assert len(pool) == expected_size, (
            f"{name}: expected {expected_size} items, got {len(pool)}"
        )
        # (2) no duplicate instructions
        qs = [q for q, _ in pool]
        if len(set(qs)) != len(qs):
            dups = [q for q, c in Counter(qs).items() if c > 1]
            raise AssertionError(f"{name}: duplicate instructions: {dups[:3]}...")
        # (3) every item is (str, str)
        for i, item in enumerate(pool):
            if not (isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[0], str) and isinstance(item[1], str)):
                raise AssertionError(f"{name}[{i}] is not a (str, str) tuple: {item!r}")

        # (4) label balance (semantic pools, levels 1+).
        # Allow |a-b| <= max(2, 10% * larger). Small absolute slack matters
        # because some labels (YES/NO in the 5-premise puzzle level) come from
        # only ~17 items, where one stray flip changes the relative ratio a lot.
        if name.startswith("semantic_") and not name.endswith("_0"):
            answers = [a for _, a in pool]
            cnt = Counter(answers)
            for pair in (("VALID", "INVALID"), ("YES", "NO")):
                if pair[0] in cnt and pair[1] in cnt:
                    a, b = cnt[pair[0]], cnt[pair[1]]
                    lo, hi = min(a, b), max(a, b)
                    if lo == 0:
                        raise AssertionError(
                            f"{name}: label {pair} has 0 count: {a} vs {b}")
                    slack = max(2, int(0.10 * hi + 0.999))
                    if (hi - lo) > slack:
                        raise AssertionError(
                            f"{name}: {pair[0]}/{pair[1]} imbalance "
                            f"{a}/{b} exceeds slack {slack}")

        # (5) regenerate and confirm the tail matches what's in the pool.
        # We extend a copy of the curated head using the same _extend_to_100
        # logic, then compare to the live pool.
        head = list(pool[:head_size])
        regen_pool = list(head)
        _extend_to_100(regen_pool, gen_fn, seed=seed)
        if regen_pool != pool:
            # Find first mismatch for a useful error message.
            for i, (live, regen) in enumerate(zip(pool, regen_pool)):
                if live != regen:
                    raise AssertionError(
                        f"{name}[{i}] regen mismatch: live={live!r} regen={regen!r}"
                    )
            raise AssertionError(f"{name}: regen length mismatch")

        if verbose:
            print(f"  {name}: OK (100 items, no dups, deterministic)")

    if verbose:
        print("All 10 T1 pools validated.")


def _maybe_validate() -> None:
    import os
    if os.environ.get("LLM_BLINK_VALIDATE") == "1":
        _validate_t1_pools(verbose=True)


_maybe_validate()


if __name__ == "__main__":
    _validate_t1_pools(verbose=True)
