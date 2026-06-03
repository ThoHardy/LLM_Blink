"""Build RSVP-style packet streams with controllable T1 load, T2 content, and lag.

Key design choices (see ../PROMPTS.md and ../../LITERATURE.md §C):
- T2 is a NOVEL random NATO-word triplet each trial -> joint-P(T2) is not at ceiling.
- T2 absolute position is held ~constant by padding before T1 (controls 'lost-in-the-middle').
- T1 has three load levels: none / easy / hard.
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
# T1 = the consolidation LOAD. Primary load type = SEMANTIC / DECISION (Thomas's choice).
# Each item is (instruction, deterministic_answer). A bank per difficulty so trials vary by seed.
# Answers are single tokens / short uppercase words -> clean to score and to teacher-force.
# ---------------------------------------------------------------------------

# easy: one-step categorisation decision
T1_SEMANTIC_EASY = [
    ("Decide whether DOLPHIN is an ANIMAL or a TOOL. Answer ANIMAL or TOOL.", "ANIMAL"),
    ("Decide whether HAMMER is an ANIMAL or a TOOL. Answer ANIMAL or TOOL.", "TOOL"),
    ("Decide whether COPPER is a METAL or a FRUIT. Answer METAL or FRUIT.", "METAL"),
    ("Decide whether MANGO is a METAL or a FRUIT. Answer METAL or FRUIT.", "FRUIT"),
    ("Decide whether VIOLIN is a VEHICLE or an INSTRUMENT. Answer VEHICLE or INSTRUMENT.",
     "INSTRUMENT"),
]

# hard: multi-step semantic/logical decision (syllogism / transitive inference)
T1_SEMANTIC_HARD = [
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

# optional arithmetic loads, kept for comparison (use loads=('none','easy_math','hard_math'))
T1_MATH = {
    "easy_math": ("Compute 2 + 3.", "5"),
    # primes in (50,70): 53,59,61,67 -> sum 240 -> *3 = 720 (verified)
    "hard_math": ("Calculate the sum of all prime numbers between 50 and 70, then multiply by 3.",
                  "720"),
}


def _get_t1(load: str, rng: random.Random):
    """Returns (t1_line, t1_answer, tag). load=='none' -> (None,None,'')."""
    if load == "none":
        return None, None, ""
    if load == "easy":
        task, ans = rng.choice(T1_SEMANTIC_EASY)
    elif load == "hard":
        task, ans = rng.choice(T1_SEMANTIC_HARD)
    elif load in T1_MATH:
        task, ans = T1_MATH[load]
    else:
        raise ValueError(f"unknown t1_load: {load}")
    return f"LOGIC OVERRIDE. {task}", ans, " - T1"


def _filler(rng: random.Random) -> str:
    return rng.choice(FILLERS).format(v=f"1.0.{rng.randint(0,9)}", n=rng.randint(1, 99))


def random_passphrase(rng: random.Random, n_words: int = 3) -> str:
    return " ".join(rng.sample(NATO, n_words))


@dataclass
class TrialConfig:
    lag: int = 2                 # packets strictly between T1 and T2
    t1_load: str = "hard"        # none | easy | hard
    regime: str = "cot"          # direct | cot
    mask: bool = False           # post-T1 mask packet
    n_pre: int | None = None     # fillers before T1 (auto: keeps T2 pos ~constant if None)
    n_post: int = 2              # fillers after T2 before End of stream
    t2_words: int = 3
    target_t2_abs_index: int = 12 # desired absolute packet index of T2 (position control)
    seed: int = 0


@dataclass
class Trial:
    system: str
    user_prefix: str            # everything up to (excluding) the T2 passphrase string
    user_full: str              # full prompt incl. T2 + template (for generation)
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

    # T1 = consolidation load (semantic/decision by default)
    t1_line, t1_ans, t1_tag = _get_t1(cfg.t1_load, rng)
    if t1_line is None:                 # 'none' load -> neutral filler in the T1 slot
        t1_line = _filler(rng)

    # position control: pad before T1 so that T2 lands at target_t2_abs_index
    # layout: [n_pre fillers][T1][mask?][lag fillers][T2]
    pre_blocks = 1 + (1 if cfg.mask else 0) + cfg.lag  # packets between first T1 and T2 (incl T1)
    if cfg.n_pre is None:
        n_pre = max(0, cfg.target_t2_abs_index - 1 - pre_blocks)
    else:
        n_pre = cfg.n_pre

    packets = []
    for _ in range(n_pre):
        packets.append(_filler(rng))
    t1_idx = len(packets) + 1
    packets.append((t1_line, t1_tag))
    if cfg.mask:
        # a mask: confusable uppercase code-words, like a fake passphrase
        packets.append(f"Reference codes logged: {random_passphrase(rng, 3)}.")
    for _ in range(cfg.lag):
        packets.append(_filler(rng))
    t2_idx = len(packets) + 1
    packets.append((f'The secure operational passphrase is "{t2}".', " - T2"))
    for _ in range(cfg.n_post):
        packets.append(_filler(rng))
    packets.append("End of stream.")

    # render packet lines
    lines = []
    for i, p in enumerate(packets, start=1):
        if isinstance(p, tuple):
            text, tag = p
            lines.append(f"[Packet {i:02d}{tag}]: {text}")
        else:
            lines.append(f"[Packet {i:02d}]: {p}")
    stream = "\n".join(lines)

    # output template + the assistant prefix that precedes the T2 slot (for logprob scoring)
    if cfg.regime == "direct":
        template = ('OUTPUT TEMPLATE:\n<Final_Answers>\n'
                    'Target 2 Result: [the exact 3-word passphrase]\n'
                    'Target 1 Result: [the exact final number]\n</Final_Answers>')
        # for direct regime the model writes T2 almost immediately:
        assistant_prefix = "<Final_Answers>\nTarget 2 Result: "
    else:  # cot
        template = ('OUTPUT TEMPLATE:\n<Thinking>\n[process the stream, do the math]\n</Thinking>\n'
                    '<Final_Answers>\nTarget 1 Result: [the exact final number]\n'
                    'Target 2 Result: [the exact 3-word passphrase]\n</Final_Answers>')
        # for logprob scoring in cot regime we score T2 conditioned on a *minimal* completed
        # thinking + T1 answer prefix (so the bottleneck of having produced T1 is represented):
        t1a = t1_ans if t1_ans is not None else "N/A"
        assistant_prefix = (f"<Thinking>\n</Thinking>\n<Final_Answers>\n"
                            f"Target 1 Result: {t1a}\nTarget 2 Result: ")

    header = f"{RULES}\nDATA STREAM:\n{stream}\n{template}"
    user_full = header
    # user_prefix is identical here; the assistant-side prefix carries the pre-T2 context
    return Trial(
        system=SYSTEM,
        user_prefix=header,
        user_full=user_full,
        t2_phrase=t2,
        t1_answer=t1_ans,
        template_prefix_after_prompt=assistant_prefix,
        config=cfg,
    )
