"""Deterministic template-based generators that expand the 10 T1 pools to 100 items each.

Each pool keeps its 5 hand-curated items frozen at the head (defined in stimuli.py).
The functions in this module produce the additional 95 items at the same difficulty
level, using seeded `random.Random` instances so output is fully reproducible.

Math answers are always computed programmatically — never hand-written.
Semantic answers are computed by the same logical procedure the LLM is being asked
to perform (transitive closure, syllogism evaluation, constraint propagation).

Only ASCII operators are used in math instructions ('+ - * /'), and instructions
end with a trailing period so they match the style of the hand-curated heads.
"""
from __future__ import annotations
import random
from typing import List, Tuple

Item = Tuple[str, str]


# ---------------------------------------------------------------------------
# Tiny prime sieve (no sympy dependency)
# ---------------------------------------------------------------------------

def _sieve(n: int) -> List[int]:
    """Return all primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


_PRIMES_2000 = _sieve(2000)
_PRIME_SET = set(_PRIMES_2000)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n <= _PRIMES_2000[-1]:
        return n in _PRIME_SET
    for p in _PRIMES_2000:
        if p * p > n:
            return True
        if n % p == 0:
            return False
    return True


def _primes_in_range(lo: int, hi: int) -> List[int]:
    """Primes p with lo <= p <= hi."""
    return [p for p in _PRIMES_2000 if lo <= p <= hi]


# ---------------------------------------------------------------------------
# MATH GENERATORS
# ---------------------------------------------------------------------------

def gen_math_l0(n: int = 95, seed: int = 100) -> List[Item]:
    """Single arithmetic op with single-digit operands; non-negative integer result."""
    rng = random.Random(seed)
    seen: set = set()
    out: List[Item] = []
    while len(out) < n:
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            res = a + b
        elif op == "-":
            a = rng.randint(2, 9)
            b = rng.randint(1, a)  # ensure non-negative
            res = a - b
        else:  # "*"
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            res = a * b
        q = f"Compute {a} {op} {b}."
        if q in seen:
            continue
        seen.add(q)
        out.append((q, str(res)))
    return out


def gen_math_l1(n: int = 95, seed: int = 101) -> List[Item]:
    """Enumerate (primes | squares | odds | multiples) in a range, then a follow-up op."""
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        kind = rng.choice(["primes_sum_mul", "primes_sum_sub",
                           "squares_sum_add", "odds_sum_sub",
                           "multiples_sum_mul"])
        if kind == "primes_sum_mul":
            lo = rng.randint(10, 80)
            hi = lo + rng.randint(15, 25)
            ps = _primes_in_range(lo, hi)
            if len(ps) < 2:
                continue
            k = rng.randint(2, 5)
            s = sum(ps)
            res = s * k
            q = (f"Calculate the sum of all prime numbers between {lo} and {hi} "
                 f"inclusive, then multiply by {k}.")
        elif kind == "primes_sum_sub":
            lo = rng.randint(10, 80)
            hi = lo + rng.randint(15, 25)
            ps = _primes_in_range(lo, hi)
            if len(ps) < 2:
                continue
            s = sum(ps)
            k = rng.randint(5, min(s - 1, 50)) if s > 6 else 1
            res = s - k
            q = (f"Calculate the sum of all prime numbers between {lo} and {hi} "
                 f"inclusive, then subtract {k}.")
        elif kind == "squares_sum_add":
            hi = rng.choice([40, 50, 60, 70, 80, 100])
            sqs = [i * i for i in range(1, int(hi ** 0.5) + 1) if i * i <= hi]
            s = sum(sqs)
            k = rng.randint(2, 30)
            res = s + k
            q = (f"Sum all perfect squares from 1 to {hi} inclusive, "
                 f"then add {k}.")
        elif kind == "odds_sum_sub":
            lo = rng.choice([1, 3, 5, 7, 9, 11, 13, 15])
            hi = lo + rng.choice([14, 16, 18, 20, 22, 24])
            # ensure lo and hi span odd numbers
            odds = [x for x in range(lo, hi + 1) if x % 2 == 1]
            s = sum(odds)
            k = rng.randint(10, max(11, s // 2))
            res = s - k
            q = (f"Sum all odd numbers from {lo} to {hi} inclusive, "
                 f"then subtract {k}.")
        else:  # multiples_sum_mul
            m = rng.choice([3, 4, 5, 6, 7])
            lo = rng.randint(1, 10)
            hi = lo + rng.randint(20, 35)
            mults = [x for x in range(lo, hi + 1) if x % m == 0]
            if len(mults) < 2:
                continue
            k = rng.randint(2, 5)
            s = sum(mults)
            res = s * k
            q = (f"Calculate the sum of all multiples of {m} from {lo} to {hi} "
                 f"inclusive, then multiply by {k}.")
        if q in seen:
            continue
        seen.add(q)
        out.append((q, str(res)))
    return out


def gen_math_l2(n: int = 95, seed: int = 102) -> List[Item]:
    """3-4 step compositions: enumerate small set + combine."""
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        kind = rng.choice(["primes_product", "factorial_sum",
                           "evens_sum_div", "square_minus_prime_count",
                           "factor_count_mul", "primes_count_squared"])
        if kind == "primes_product":
            lo = rng.randint(10, 40)
            hi = lo + rng.randint(8, 14)
            ps = _primes_in_range(lo, hi)
            if len(ps) != 2:
                continue
            res = ps[0] * ps[1]
            q = (f"Find all prime numbers between {lo} and {hi} inclusive. "
                 f"Multiply them together.")
        elif kind == "factorial_sum":
            a = rng.randint(2, 5)
            b = rng.randint(a + 1, 6)
            fa = 1
            for i in range(1, a + 1):
                fa *= i
            fb = 1
            for i in range(1, b + 1):
                fb *= i
            res = fa + fb
            q = f"Compute {a} factorial plus {b} factorial."
        elif kind == "evens_sum_div":
            lo = 2
            hi = rng.choice([12, 14, 16, 18, 20, 22, 24])
            evens = [x for x in range(lo, hi + 1) if x % 2 == 0]
            s = sum(evens)
            # pick a divisor that divides s
            divs = [d for d in (2, 3, 4, 5, 6) if s % d == 0]
            if not divs:
                continue
            d = rng.choice(divs)
            res = s // d
            q = (f"Sum all even numbers from {lo} to {hi} inclusive, "
                 f"then divide by {d}.")
        elif kind == "square_minus_prime_count":
            base = rng.randint(8, 20)
            cap = rng.choice([20, 25, 30, 40, 50])
            cnt = len([p for p in _PRIMES_2000 if p < cap])
            res = base * base - cnt
            q = (f"What is {base} squared minus the count of prime numbers "
                 f"less than {cap}?")
        elif kind == "factor_count_mul":
            target = rng.choice([24, 30, 36, 40, 48, 60, 72])
            factors = [d for d in range(1, target + 1) if target % d == 0]
            cnt = len(factors)
            k = rng.randint(3, 9)
            res = cnt * k
            q = (f"List all factors of {target}. How many are there? "
                 f"Multiply that count by {k}.")
        else:  # primes_count_squared
            cap = rng.choice([15, 20, 25, 30, 40, 50])
            cnt = len([p for p in _PRIMES_2000 if p < cap])
            res = cnt * cnt
            q = (f"How many prime numbers are strictly less than {cap}? "
                 f"Square that count.")
        if q in seen:
            continue
        seen.add(q)
        out.append((q, str(res)))
    return out


def gen_math_l3(n: int = 95, seed: int = 103) -> List[Item]:
    """4-5 steps: two enumerations or modular/divisibility filters."""
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        kind = rng.choice(["prime_sum_mod", "two_digit_squares_minus_cube",
                           "start_minus_primes", "div_a_or_b_not_both",
                           "largest_two_digit_prime_mod", "sum_div_x_in_range_mod"])
        if kind == "prime_sum_mod":
            hi = rng.choice([20, 25, 30, 40, 50])
            ps = [p for p in _PRIMES_2000 if 1 <= p <= hi]
            s = sum(ps)
            m = rng.choice([7, 9, 11, 13])
            res = s % m
            q = (f"Sum all prime numbers from 1 to {hi} inclusive. "
                 f"What is the remainder when that sum is divided by {m}?")
        elif kind == "two_digit_squares_minus_cube":
            sqs = [i * i for i in range(4, 10)]  # 16..81
            s = sum(sqs)
            base = rng.randint(2, 6)
            res = s - base ** 3
            q = (f"Compute the sum of all two-digit perfect squares, "
                 f"then subtract {base} cubed.")
        elif kind == "start_minus_primes":
            cap = rng.choice([15, 20, 25, 30])
            ps = [p for p in _PRIMES_2000 if p < cap]
            start = rng.randint(sum(ps) + 5, sum(ps) + 80)
            res = start - sum(ps)
            q = (f"Start with {start}. Subtract every prime number less than {cap} "
                 f"in increasing order. What is the result?")
        elif kind == "div_a_or_b_not_both":
            a = rng.choice([2, 3, 4, 5])
            b = rng.choice([3, 4, 5, 6, 7, 9])
            if a == b:
                continue
            top = rng.choice([40, 50, 60, 80, 100])
            cnt = sum(1 for x in range(1, top + 1)
                      if (x % a == 0) ^ (x % b == 0))
            res = cnt
            q = (f"How many integers from 1 to {top} are divisible by either "
                 f"{a} or {b}, but not by both {a} and {b}?")
        elif kind == "largest_two_digit_prime_mod":
            m = rng.choice([4, 5, 6, 7, 8])
            r = rng.randint(1, m - 1)
            cands = [p for p in _PRIMES_2000 if 10 <= p <= 99 and p % m == r]
            if not cands:
                continue
            res = max(cands)
            q = (f"What is the largest two-digit prime number that leaves a "
                 f"remainder of {r} when divided by {m}?")
        else:  # sum_div_x_in_range_mod
            x = rng.choice([3, 4, 5, 6, 7])
            top = rng.choice([30, 40, 50, 60, 80])
            mults = [v for v in range(1, top + 1) if v % x == 0]
            s = sum(mults)
            m = rng.choice([7, 9, 11, 13])
            res = s % m
            q = (f"Sum all multiples of {x} from 1 to {top} inclusive. "
                 f"What is the remainder when that sum is divided by {m}?")
        if q in seen:
            continue
        seen.add(q)
        out.append((q, str(res)))
    return out


def gen_math_l4(n: int = 95, seed: int = 104) -> List[Item]:
    """5+ steps: nested enumeration, digit-sum + modular constraints, filtered aggregates."""
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    while len(out) < n and attempts < n * 80:
        attempts += 1
        kind = rng.choice(["sum_prime_squares", "digit_sum_odd_multiple",
                           "div_a_not_b_not_c_mul", "iterated_double_add",
                           "two_digit_squares_extrema", "filtered_sum_minus_count_cubed",
                           "count_primes_digit_sum"])
        if kind == "sum_prime_squares":
            cap = rng.choice([15, 20, 25, 30])
            ps = [p for p in _PRIMES_2000 if p < cap]
            res = sum(p * p for p in ps)
            q = (f"Compute the sum of the squares of all prime numbers less than {cap}.")
        elif kind == "digit_sum_odd_multiple":
            m = rng.choice([7, 9, 11, 13])
            lo = rng.choice([100, 150, 200, 250])
            hi = lo + rng.choice([100, 150])
            ds_target = rng.randint(7, 16)
            cands = [n_ for n_ in range(lo + 1, hi)
                     if n_ % m == 0 and n_ % 2 == 1
                     and sum(int(c) for c in str(n_)) == ds_target]
            if len(cands) != 1:
                continue
            res = cands[0]
            q = (f"N is an odd multiple of {m} strictly between {lo} and {hi}. "
                 f"The sum of N's digits equals {ds_target}. What is N?")
        elif kind == "div_a_not_b_not_c_mul":
            a = rng.choice([2, 3, 4])
            b, c = rng.sample([3, 5, 7, 9], 2)
            if b == a or c == a:
                continue
            top = rng.choice([50, 80, 100, 120])
            cnt = sum(1 for x in range(1, top + 1)
                      if x % a == 0 and x % b != 0 and x % c != 0)
            k = rng.randint(2, 6)
            res = cnt * k
            q = (f"How many integers from 1 to {top} are divisible by {a} but "
                 f"not divisible by {b} and not divisible by {c}? "
                 f"Multiply that count by {k}.")
        elif kind == "iterated_double_add":
            start = rng.randint(1, 5)
            add = rng.randint(1, 5)
            steps = rng.randint(4, 6)
            v = start
            for _ in range(steps):
                v = v * 2 + add
            res = v
            q = (f"Start with {start}. Repeat {steps} times: double the current "
                 f"number and add {add}. What is the final result?")
        elif kind == "two_digit_squares_extrema":
            sqs = [i * i for i in range(4, 10)]  # 16..81
            small, large = min(sqs), max(sqs)
            cnt = len(sqs)
            mode = rng.choice(["mul_add_count", "mul_sub_count"])
            if mode == "mul_add_count":
                res = small * large + cnt
                q = (f"Find all two-digit perfect squares. Multiply the largest "
                     f"by the smallest, then add the total count of such numbers.")
            else:
                res = small * large - cnt
                q = (f"Find all two-digit perfect squares. Multiply the largest "
                     f"by the smallest, then subtract the total count of such numbers.")
        elif kind == "filtered_sum_minus_count_cubed":
            x = rng.choice([3, 4, 5, 6])
            top = rng.choice([30, 40, 50, 60])
            mults = [v for v in range(1, top + 1) if v % x == 0]
            s = sum(mults)
            cnt = len(mults)
            res = s - cnt ** 3
            q = (f"Sum all multiples of {x} from 1 to {top} inclusive. "
                 f"Count how many multiples there were and cube that count. "
                 f"Subtract the cube from the sum.")
        else:  # count_primes_digit_sum
            lo = rng.choice([20, 30, 40, 50, 60])
            hi = lo + rng.randint(40, 80)
            ds_target = rng.randint(5, 13)
            ps = [p for p in _PRIMES_2000 if lo <= p <= hi
                  and sum(int(c) for c in str(p)) == ds_target]
            cnt = len(ps)
            k = rng.randint(2, 6)
            res = cnt * k
            q = (f"Count the prime numbers between {lo} and {hi} inclusive "
                 f"whose digits sum to {ds_target}. Multiply that count by {k}.")
        if q in seen:
            continue
        seen.add(q)
        out.append((q, str(res)))
    return out


# ---------------------------------------------------------------------------
# Semantic generators — shared resources
# ---------------------------------------------------------------------------

NONCE = [
    "gleeps", "morks", "florks", "blims", "vorps", "wugs", "zorps", "snorfs",
    "trens", "wumps", "trils", "snicks", "krants", "flurbs", "drovs", "pims",
    "ralks", "frans", "blorgs", "klemps", "snigs", "wozzles", "yarps", "quibs",
    "plomps", "thrups", "vails", "grints", "skrans", "drabs", "plonks", "vrils",
    "snools", "mibs", "trels", "grebs", "knifs", "plurms", "smegs", "torps",
]

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
         "Iris", "Jack", "Kira", "Leo", "Mona", "Nick", "Olive", "Pia",
         "Quinn", "Ravi", "Sara", "Tom"]

# Real categorisation items: (item, real_category, distractor_category)
CAT_ITEMS = [
    # animal / tool
    ("DOG", "ANIMAL", "TOOL"), ("CAT", "ANIMAL", "TOOL"),
    ("ELEPHANT", "ANIMAL", "TOOL"), ("RABBIT", "ANIMAL", "TOOL"),
    ("LION", "ANIMAL", "TOOL"), ("TIGER", "ANIMAL", "TOOL"),
    ("EAGLE", "ANIMAL", "TOOL"), ("WHALE", "ANIMAL", "TOOL"),
    ("HORSE", "ANIMAL", "TOOL"), ("FROG", "ANIMAL", "TOOL"),
    ("SNAKE", "ANIMAL", "TOOL"), ("BEAR", "ANIMAL", "TOOL"),
    ("WOLF", "ANIMAL", "TOOL"), ("DEER", "ANIMAL", "TOOL"),
    ("FOX", "ANIMAL", "TOOL"), ("OWL", "ANIMAL", "TOOL"),
    ("SHARK", "ANIMAL", "TOOL"), ("OCTOPUS", "ANIMAL", "TOOL"),
    ("BUTTERFLY", "ANIMAL", "TOOL"), ("MOUSE", "ANIMAL", "TOOL"),
    ("WRENCH", "TOOL", "ANIMAL"), ("SCREWDRIVER", "TOOL", "ANIMAL"),
    ("SAW", "TOOL", "ANIMAL"), ("DRILL", "TOOL", "ANIMAL"),
    ("PLIERS", "TOOL", "ANIMAL"), ("CHISEL", "TOOL", "ANIMAL"),
    ("AXE", "TOOL", "ANIMAL"), ("SHOVEL", "TOOL", "ANIMAL"),
    ("RAKE", "TOOL", "ANIMAL"), ("MALLET", "TOOL", "ANIMAL"),
    ("CLAMP", "TOOL", "ANIMAL"), ("SCISSORS", "TOOL", "ANIMAL"),
    ("CROWBAR", "TOOL", "ANIMAL"), ("HOE", "TOOL", "ANIMAL"),

    # metal / fruit
    ("IRON", "METAL", "FRUIT"), ("GOLD", "METAL", "FRUIT"),
    ("SILVER", "METAL", "FRUIT"), ("PLATINUM", "METAL", "FRUIT"),
    ("NICKEL", "METAL", "FRUIT"), ("ZINC", "METAL", "FRUIT"),
    ("TIN", "METAL", "FRUIT"), ("LEAD", "METAL", "FRUIT"),
    ("ALUMINUM", "METAL", "FRUIT"), ("TITANIUM", "METAL", "FRUIT"),
    ("BRASS", "METAL", "FRUIT"), ("STEEL", "METAL", "FRUIT"),
    ("APPLE", "FRUIT", "METAL"), ("BANANA", "FRUIT", "METAL"),
    ("ORANGE", "FRUIT", "METAL"), ("GRAPE", "FRUIT", "METAL"),
    ("PEAR", "FRUIT", "METAL"), ("PEACH", "FRUIT", "METAL"),
    ("PLUM", "FRUIT", "METAL"), ("CHERRY", "FRUIT", "METAL"),
    ("LEMON", "FRUIT", "METAL"), ("LIME", "FRUIT", "METAL"),
    ("PAPAYA", "FRUIT", "METAL"), ("KIWI", "FRUIT", "METAL"),
    ("PINEAPPLE", "FRUIT", "METAL"), ("APRICOT", "FRUIT", "METAL"),

    # instrument / vehicle
    ("PIANO", "INSTRUMENT", "VEHICLE"), ("GUITAR", "INSTRUMENT", "VEHICLE"),
    ("FLUTE", "INSTRUMENT", "VEHICLE"), ("DRUM", "INSTRUMENT", "VEHICLE"),
    ("TRUMPET", "INSTRUMENT", "VEHICLE"), ("CELLO", "INSTRUMENT", "VEHICLE"),
    ("HARP", "INSTRUMENT", "VEHICLE"), ("CLARINET", "INSTRUMENT", "VEHICLE"),
    ("SAXOPHONE", "INSTRUMENT", "VEHICLE"), ("ORGAN", "INSTRUMENT", "VEHICLE"),
    ("BANJO", "INSTRUMENT", "VEHICLE"), ("OBOE", "INSTRUMENT", "VEHICLE"),
    ("BICYCLE", "VEHICLE", "INSTRUMENT"), ("CAR", "VEHICLE", "INSTRUMENT"),
    ("TRUCK", "VEHICLE", "INSTRUMENT"), ("MOTORCYCLE", "VEHICLE", "INSTRUMENT"),
    ("BUS", "VEHICLE", "INSTRUMENT"), ("AIRPLANE", "VEHICLE", "INSTRUMENT"),
    ("HELICOPTER", "VEHICLE", "INSTRUMENT"), ("BOAT", "VEHICLE", "INSTRUMENT"),
    ("SCOOTER", "VEHICLE", "INSTRUMENT"), ("TRAIN", "VEHICLE", "INSTRUMENT"),
    ("TRAM", "VEHICLE", "INSTRUMENT"), ("VAN", "VEHICLE", "INSTRUMENT"),

    # planet / country
    ("MARS", "PLANET", "COUNTRY"), ("JUPITER", "PLANET", "COUNTRY"),
    ("SATURN", "PLANET", "COUNTRY"), ("VENUS", "PLANET", "COUNTRY"),
    ("MERCURY", "PLANET", "COUNTRY"), ("NEPTUNE", "PLANET", "COUNTRY"),
    ("URANUS", "PLANET", "COUNTRY"),
    ("FRANCE", "COUNTRY", "PLANET"), ("BRAZIL", "COUNTRY", "PLANET"),
    ("JAPAN", "COUNTRY", "PLANET"), ("CANADA", "COUNTRY", "PLANET"),
    ("EGYPT", "COUNTRY", "PLANET"), ("KENYA", "COUNTRY", "PLANET"),
    ("SWEDEN", "COUNTRY", "PLANET"), ("PORTUGAL", "COUNTRY", "PLANET"),
    ("THAILAND", "COUNTRY", "PLANET"), ("MEXICO", "COUNTRY", "PLANET"),
    ("NORWAY", "COUNTRY", "PLANET"), ("AUSTRALIA", "COUNTRY", "PLANET"),

    # vegetable / gemstone
    ("CARROT", "VEGETABLE", "GEMSTONE"), ("POTATO", "VEGETABLE", "GEMSTONE"),
    ("ONION", "VEGETABLE", "GEMSTONE"), ("SPINACH", "VEGETABLE", "GEMSTONE"),
    ("BROCCOLI", "VEGETABLE", "GEMSTONE"), ("CABBAGE", "VEGETABLE", "GEMSTONE"),
    ("LETTUCE", "VEGETABLE", "GEMSTONE"), ("CELERY", "VEGETABLE", "GEMSTONE"),
    ("CUCUMBER", "VEGETABLE", "GEMSTONE"), ("PEPPER", "VEGETABLE", "GEMSTONE"),
    ("RUBY", "GEMSTONE", "VEGETABLE"), ("DIAMOND", "GEMSTONE", "VEGETABLE"),
    ("EMERALD", "GEMSTONE", "VEGETABLE"), ("SAPPHIRE", "GEMSTONE", "VEGETABLE"),
    ("OPAL", "GEMSTONE", "VEGETABLE"), ("AMETHYST", "GEMSTONE", "VEGETABLE"),
    ("TOPAZ", "GEMSTONE", "VEGETABLE"), ("GARNET", "GEMSTONE", "VEGETABLE"),
    ("JADE", "GEMSTONE", "VEGETABLE"), ("PEARL", "GEMSTONE", "VEGETABLE"),

    # bird / building
    ("SPARROW", "BIRD", "BUILDING"), ("ROBIN", "BIRD", "BUILDING"),
    ("FALCON", "BIRD", "BUILDING"), ("PARROT", "BIRD", "BUILDING"),
    ("PIGEON", "BIRD", "BUILDING"), ("PENGUIN", "BIRD", "BUILDING"),
    ("FLAMINGO", "BIRD", "BUILDING"), ("HAWK", "BIRD", "BUILDING"),
    ("CHURCH", "BUILDING", "BIRD"), ("MOSQUE", "BUILDING", "BIRD"),
    ("LIBRARY", "BUILDING", "BIRD"), ("HOSPITAL", "BUILDING", "BIRD"),
    ("CASTLE", "BUILDING", "BIRD"), ("MUSEUM", "BUILDING", "BIRD"),
    ("THEATRE", "BUILDING", "BIRD"), ("FACTORY", "BUILDING", "BIRD"),

    # river / cereal
    ("AMAZON", "RIVER", "CEREAL"), ("NILE", "RIVER", "CEREAL"),
    ("DANUBE", "RIVER", "CEREAL"), ("THAMES", "RIVER", "CEREAL"),
    ("GANGES", "RIVER", "CEREAL"), ("MEKONG", "RIVER", "CEREAL"),
    ("WHEAT", "CEREAL", "RIVER"), ("BARLEY", "CEREAL", "RIVER"),
    ("OAT", "CEREAL", "RIVER"), ("RYE", "CEREAL", "RIVER"),
    ("MILLET", "CEREAL", "RIVER"), ("SORGHUM", "CEREAL", "RIVER"),
]


def _pick_nonces(rng: random.Random, k: int) -> List[str]:
    return rng.sample(NONCE, k)


def _pick_names(rng: random.Random, k: int) -> List[str]:
    return rng.sample(NAMES, k)


# Article helper for clean grammar
def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


# ---------------------------------------------------------------------------
# SEMANTIC LEVEL 0 — one-step categorisation
# ---------------------------------------------------------------------------

def gen_semantic_l0(n: int = 95, seed: int = 200) -> List[Item]:
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    # We want balanced "real-is-on-left" vs "real-is-on-right" position.
    # Build a list of items in a shuffled but position-balanced order.
    items_pool = list(CAT_ITEMS)
    rng.shuffle(items_pool)
    # cycle through with index parity controlling order
    idx = 0
    left_right_flip = 0
    while len(out) < n:
        item, real, distractor = items_pool[idx % len(items_pool)]
        idx += 1
        if left_right_flip % 2 == 0:
            c1, c2 = real, distractor
        else:
            c1, c2 = distractor, real
        left_right_flip += 1
        art1 = _article(c1)
        art2 = _article(c2)
        q = (f"Decide whether {item} is {art1} {c1} or {art2} {c2}. "
             f"Answer {c1} or {c2}.")
        if q in seen:
            continue
        seen.add(q)
        out.append((q, real))
    return out


# ---------------------------------------------------------------------------
# SEMANTIC LEVEL 1 — 2-premise syllogism (Barbara valid, some-some invalid)
# or 3-element transitive
# ---------------------------------------------------------------------------

# Transitive relation triples: (rel_word, q_rel_word, opposite_q_rel_word)
# Used like "A is RELWORD than B."
REL_PAIRS = [
    ("heavier", "heavier", "lighter"),
    ("lighter", "lighter", "heavier"),
    ("taller", "taller", "shorter"),
    ("shorter", "shorter", "taller"),
    ("older", "older", "younger"),
    ("younger", "younger", "older"),
    ("faster", "faster", "slower"),
    ("slower", "slower", "faster"),
]


def _syllogism_barbara(rng: random.Random) -> Item:
    """All X are Y. All Y are Z. -> All X are Z?  VALID."""
    x, y, z = _pick_nonces(rng, 3)
    q = (f"All {x} are {y}. All {y} are {z}. "
         f"Does it follow that all {x} are {z}? Answer VALID or INVALID.")
    return q, "VALID"


def _syllogism_some_some(rng: random.Random) -> Item:
    """All X are Y. Some Y are Z. -> Some X are Z?  INVALID."""
    x, y, z = _pick_nonces(rng, 3)
    q = (f"All {x} are {y}. Some {y} are {z}. "
         f"Does it follow that some {x} are {z}? Answer VALID or INVALID.")
    return q, "INVALID"


def _transitive_3(rng: random.Random) -> Item:
    """A REL B, B REL C, is A REL C? YES (or NO if question uses opposite rel)."""
    a, b, c = _pick_names(rng, 3)
    rel_word, q_rel, opp_rel = rng.choice(REL_PAIRS)
    use_opposite = rng.random() < 0.5
    if use_opposite:
        q = (f"{a} is {rel_word} than {b}. {b} is {rel_word} than {c}. "
             f"Is {a} {opp_rel} than {c}? Answer YES or NO.")
        ans = "NO"
    else:
        q = (f"{a} is {rel_word} than {b}. {b} is {rel_word} than {c}. "
             f"Is {a} {q_rel} than {c}? Answer YES or NO.")
        ans = "YES"
    return q, ans


def gen_semantic_l1(n: int = 95, seed: int = 201) -> List[Item]:
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    # Target ~equal counts across 3 templates; balance VALID/INVALID, YES/NO.
    # We aim for ~32 each; loop with constraints.
    attempts = 0
    target_each = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    counts = [0, 0, 0]
    while sum(counts) < n and attempts < n * 100:
        attempts += 1
        # Pick the under-represented bucket
        slot = min(range(3), key=lambda i: counts[i] - target_each[i] * 100)
        # but prefer the one most under target
        under = [i for i in range(3) if counts[i] < target_each[i]]
        if under:
            slot = rng.choice(under)
        if slot == 0:
            q, ans = _syllogism_barbara(rng)
        elif slot == 1:
            q, ans = _syllogism_some_some(rng)
        else:
            q, ans = _transitive_3(rng)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, ans))
        counts[slot] += 1
    return out


# ---------------------------------------------------------------------------
# SEMANTIC LEVEL 2 — Celarent / Darii / Ferio trap / 4-elt transitive
# ---------------------------------------------------------------------------

def _celarent(rng: random.Random) -> Item:
    """All M are G. No G are S. -> No M are S?  VALID."""
    m, g, s = _pick_nonces(rng, 3)
    q = (f"All {m} are {g}. No {g} are {s}. "
         f"Does it follow that no {m} are {s}? Answer VALID or INVALID.")
    return q, "VALID"


def _darii(rng: random.Random) -> Item:
    """Some V are B. All B are T. -> Some V are T?  VALID."""
    v, b, t = _pick_nonces(rng, 3)
    q = (f"Some {v} are {b}. All {b} are {t}. "
         f"Does it follow that some {v} are {t}? Answer VALID or INVALID.")
    return q, "VALID"


def _ferio_trap(rng: random.Random) -> Item:
    """All F are W. Some W are not G. -> Some F are not G?  INVALID (illicit major)."""
    f, w, g = _pick_nonces(rng, 3)
    q = (f"All {f} are {w}. Some {w} are not {g}. "
         f"Does it follow that some {f} are not {g}? Answer VALID or INVALID.")
    return q, "INVALID"


def _transitive_4(rng: random.Random) -> Item:
    """A>B>C>D via REL. Question on a non-adjacent pair, e.g. is A REL D?"""
    a, b, c, d = _pick_names(rng, 4)
    rel_word, q_rel, opp_rel = rng.choice(REL_PAIRS)
    # ordering A>B>C>D under "REL"
    use_opposite = rng.random() < 0.5
    # pick a non-adjacent pair: (a,c), (a,d), (b,d)
    pair = rng.choice([(a, c), (a, d), (b, d)])
    p1, p2 = pair
    if use_opposite:
        q = (f"{a} is {rel_word} than {b}. {b} is {rel_word} than {c}. "
             f"{c} is {rel_word} than {d}. "
             f"Is {p1} {opp_rel} than {p2}? Answer YES or NO.")
        ans = "NO"
    else:
        q = (f"{a} is {rel_word} than {b}. {b} is {rel_word} than {c}. "
             f"{c} is {rel_word} than {d}. "
             f"Is {p1} {q_rel} than {p2}? Answer YES or NO.")
        ans = "YES"
    return q, ans


def gen_semantic_l2(n: int = 95, seed: int = 202) -> List[Item]:
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    target_each = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    counts = [0, 0, 0, 0]
    gens = [_celarent, _darii, _ferio_trap, _transitive_4]
    while sum(counts) < n and attempts < n * 100:
        attempts += 1
        under = [i for i in range(4) if counts[i] < target_each[i]]
        slot = rng.choice(under) if under else rng.randint(0, 3)
        q, ans = gens[slot](rng)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, ans))
        counts[slot] += 1
    return out


# ---------------------------------------------------------------------------
# SEMANTIC LEVEL 3 — 4-step transitive (5 entities) or 3-premise chains with distractor
# ---------------------------------------------------------------------------

def _transitive_5(rng: random.Random) -> Item:
    """5-entity chain, question on non-adjacent pair (often extremes)."""
    e = _pick_names(rng, 5)
    rel_word, q_rel, opp_rel = rng.choice(REL_PAIRS)
    use_opposite = rng.random() < 0.5
    # Pick a non-adjacent pair (gap >= 2 in 0..4)
    options = [(i, j) for i in range(5) for j in range(i + 2, 5)]
    i, j = rng.choice(options)
    p1, p2 = e[i], e[j]
    premises = " ".join(f"{e[k]} is {rel_word} than {e[k+1]}." for k in range(4))
    if use_opposite:
        q = (f"{premises} Is {p1} {opp_rel} than {p2}? Answer YES or NO.")
        ans = "NO"
    else:
        q = (f"{premises} Is {p1} {q_rel} than {p2}? Answer YES or NO.")
        ans = "YES"
    return q, ans


def _chain_all_all_no(rng: random.Random) -> Item:
    """All V are B. All B are S. No S are G. -> No V are G?  VALID."""
    v, b, s, g = _pick_nonces(rng, 4)
    q = (f"All {v} are {b}. All {b} are {s}. No {s} are {g}. "
         f"Does it follow that no {v} are {g}? Answer VALID or INVALID.")
    return q, "VALID"


def _chain_some_invalid(rng: random.Random) -> Item:
    """All G are F. Some F are S. All S are W. -> Some G are W?  INVALID."""
    g, f, s, w = _pick_nonces(rng, 4)
    q = (f"All {g} are {f}. Some {f} are {s}. All {s} are {w}. "
         f"Does it follow that some {g} are {w}? Answer VALID or INVALID.")
    return q, "INVALID"


def _chain_some_all_no_valid(rng: random.Random) -> Item:
    """Some W are Z. All Z are M. No M are T. -> Some W are not T?  VALID."""
    w, z, m, t = _pick_nonces(rng, 4)
    q = (f"Some {w} are {z}. All {z} are {m}. No {m} are {t}. "
         f"Does it follow that some {w} are not {t}? Answer VALID or INVALID.")
    return q, "VALID"


def gen_semantic_l3(n: int = 95, seed: int = 203) -> List[Item]:
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    target_each = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    counts = [0, 0, 0, 0]
    gens = [_transitive_5, _chain_all_all_no, _chain_some_invalid, _chain_some_all_no_valid]
    while sum(counts) < n and attempts < n * 100:
        attempts += 1
        under = [i for i in range(4) if counts[i] < target_each[i]]
        slot = rng.choice(under) if under else rng.randint(0, 3)
        q, ans = gens[slot](rng)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, ans))
        counts[slot] += 1
    return out


# ---------------------------------------------------------------------------
# SEMANTIC LEVEL 4 — 5-premise syllogisms, 5-step transitive (6 entities),
# constraint-based ordering with 5 entities
# ---------------------------------------------------------------------------

def _transitive_6(rng: random.Random) -> Item:
    """6-entity chain, question on non-adjacent pair."""
    e = _pick_names(rng, 6)
    rel_word, q_rel, opp_rel = rng.choice(REL_PAIRS)
    use_opposite = rng.random() < 0.5
    options = [(i, j) for i in range(6) for j in range(i + 2, 6)]
    i, j = rng.choice(options)
    p1, p2 = e[i], e[j]
    premises = " ".join(f"{e[k]} is {rel_word} than {e[k+1]}." for k in range(5))
    if use_opposite:
        q = (f"{premises} Is {p1} {opp_rel} than {p2}? Answer YES or NO.")
        ans = "NO"
    else:
        q = (f"{premises} Is {p1} {q_rel} than {p2}? Answer YES or NO.")
        ans = "YES"
    return q, ans


def _five_premise_chain_valid(rng: random.Random) -> Item:
    """Z->M->S, No S->F, distractor premises about F->W and some W are Z.

    Z->M->S means All Z are M, All M are S.  No S are F.  Distractors: All F are W,
    Some W are Z (these add noise but don't break the conclusion "No Z are F").
    """
    z, m, s, f, w = _pick_nonces(rng, 5)
    q = (f"All {z} are {m}. All {m} are {s}. No {s} are {f}. "
         f"All {f} are {w}. Some {w} are {z}. "
         f"Does it follow that no {z} are {f}? Answer VALID or INVALID.")
    return q, "VALID"


def _five_premise_chain_invalid(rng: random.Random) -> Item:
    """All V->B, All S->G, Some B are S, No G->W, All W->T.

    From "All V are B" and "Some B are S" we cannot conclude "Some V are S"
    (subject middle-term is undistributed). Even with All S are G and No G are W,
    we cannot conclude "No V are W". INVALID.
    """
    v, b, s, g, w, t = _pick_nonces(rng, 6)
    q = (f"All {v} are {b}. All {s} are {g}. Some {b} are {s}. "
         f"No {g} are {w}. All {w} are {t}. "
         f"Does it follow that no {v} are {w}? Answer VALID or INVALID.")
    return q, "INVALID"


def _five_premise_some_not_valid(rng: random.Random) -> Item:
    """Some G are B. All B are S. Some S are V. All V are W. No W are G.
    Distractor: 'Some S are V' is loose. From "No W are G" and "All V are W"
    we get "No V are G". Conclusion 'some G are not W'?
    G overlaps B; All B are S, but B might map into W via S... we need a clean valid.

    Use: Some G are B. All B are S. All S are V. No V are W. (4 needed premises)
    + distractor 'All W are T'. Conclusion: 'some G are not W'?
    Some G are B (exists g0 in B); g0 in B subset S subset V; V disjoint W
    => g0 not in W. So 'some G are not W' VALID.
    """
    g, b, s, v, w, t = _pick_nonces(rng, 6)
    q = (f"Some {g} are {b}. All {b} are {s}. All {s} are {v}. "
         f"No {v} are {w}. All {w} are {t}. "
         f"Does it follow that some {g} are not {w}? Answer VALID or INVALID.")
    return q, "VALID"


def _five_premise_existential_invalid(rng: random.Random) -> Item:
    """Some A are B. Some B are C. Some C are D. All D are E. No E are F.
    Distractor: All F are G. Conclusion: 'some A are not F'?

    'Some A are B' + 'Some B are C' doesn't give 'Some A are C' (chained
    existentials are invalid).  Cannot conclude 'some A are not F'. INVALID.
    """
    a, b, c, d, e, f, g = _pick_nonces(rng, 7)
    q = (f"Some {a} are {b}. Some {b} are {c}. Some {c} are {d}. "
         f"All {d} are {e}. No {e} are {f}. "
         f"Does it follow that some {a} are not {f}? Answer VALID or INVALID.")
    return q, "INVALID"


def _constraint_puzzle_5(rng: random.Random) -> Item:
    """5-entity ordering puzzle with 4-5 constraints; unique solution; asks for k-th place.

    Solver: enumerate all permutations of 5 entities, filter by constraints,
    require exactly one solution.
    """
    from itertools import permutations
    entities = rng.sample(["P", "Q", "R", "S", "T", "U", "V", "W"], 5)
    rel_word, _, _ = rng.choice(REL_PAIRS)
    # Try until we find a constraint set that yields a unique solution.
    for _ in range(80):
        constraints = []
        constraint_texts = []
        # Generate 4 random "X before Y" constraints (where ordering = a permutation)
        n_constraints = rng.choice([4, 5])
        c_set = set()
        for _ in range(n_constraints * 3):
            x, y = rng.sample(entities, 2)
            if (x, y) in c_set or (y, x) in c_set:
                continue
            c_set.add((x, y))
            constraints.append((x, y))
            constraint_texts.append(f"{x} finished before {y}.")
            if len(constraints) >= n_constraints:
                break
        if len(constraints) < n_constraints:
            continue
        # Solve: an ordering is a permutation; "X before Y" means X earlier than Y
        solutions = []
        for perm in permutations(entities):
            pos = {e: i for i, e in enumerate(perm)}
            ok = all(pos[x] < pos[y] for x, y in constraints)
            if ok:
                solutions.append(perm)
                if len(solutions) > 1:
                    break
        if len(solutions) != 1:
            continue
        sol = solutions[0]
        k = rng.choice([1, 2, 3, 4, 5])
        ord_word = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[k]
        ans = sol[k - 1]
        ent_list = ", ".join(entities[:-1]) + ", and " + entities[-1]
        cs = " ".join(constraint_texts)
        q = (f"There are five runners: {ent_list}. "
             f"{cs} "
             f"Who finished in {ord_word} place? "
             f"Answer {entities[0]}, {entities[1]}, {entities[2]}, "
             f"{entities[3]}, or {entities[4]}.")
        return q, ans
    # Fallback: trivial chain
    a, b, c, d, e = entities
    ans = c
    q = (f"There are five runners: {a}, {b}, {c}, {d}, and {e}. "
         f"{a} finished before {b}. {b} finished before {c}. "
         f"{c} finished before {d}. {d} finished before {e}. "
         f"Who finished in third place? Answer {a}, {b}, {c}, {d}, or {e}.")
    return q, ans


def gen_semantic_l4(n: int = 95, seed: int = 204) -> List[Item]:
    rng = random.Random(seed)
    out: List[Item] = []
    seen: set = set()
    attempts = 0
    # 6 generator slots; aim for balanced VALID/INVALID / YES/NO / puzzle
    gens = [
        _transitive_6,
        _five_premise_chain_valid,
        _five_premise_chain_invalid,
        _five_premise_some_not_valid,
        _five_premise_existential_invalid,
        _constraint_puzzle_5,
    ]
    target_each = [n // len(gens) + (1 if i < n % len(gens) else 0)
                   for i in range(len(gens))]
    counts = [0] * len(gens)
    while sum(counts) < n and attempts < n * 100:
        attempts += 1
        under = [i for i in range(len(gens)) if counts[i] < target_each[i]]
        slot = rng.choice(under) if under else rng.randint(0, len(gens) - 1)
        q, ans = gens[slot](rng)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, ans))
        counts[slot] += 1
    return out
