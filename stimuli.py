"""Build RSVP-style packet streams with controllable T1 load, T2 content, and lag.

Key design choices (see ../PROMPTS.md and ../LITERATURE.md §C):
- T2 is a NOVEL random NATO-word triplet each trial -> joint-P(T2) is not at ceiling.
- Stream length is held constant at total_packets (default 15); n_post (fillers after T2)
  is random by default and bounded by the lag, n_pre absorbs the remainder.
- T1 has five load levels per track: semantic_0..4 and math_0..4, plus two baselines:
  'trivial' (a tagged T1 packet that demands no computation — schema-preserving) and
  'none' (untagged filler in the T1 slot — NOTE: schema-violating, the stream then has
  no '[Packet xx - T1]' while the template still asks for 'Target 1 Result:').
  Aliases: easy=semantic_0, hard=semantic_1, easy_math=math_0, hard_math=math_1.
- Optional post-T1 'mask' packet (the human AB needs a mask).
- Two output regimes: 'direct' (answer T2 first, no scratchpad) and 'cot' (solve T1 first).
- Combined A×B×H design (2026-07-20): TrialConfig.n_tasks=int switches to task streams
  (n_tasks named tasks incl. the passphrase, free report, hidden count) — see §'Combined
  A×B×H design' below. n_tasks=None (default) keeps the legacy design above BIT-EXACT.
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
# NOTE: FILLERS above is kept for reference/back-compat; _filler() now samples
# from the deterministic FILLER_POOL built below (Item 7).


# ---------------------------------------------------------------------------
# Deterministic 1000-phrase filler pool (Item 7).
#
# Generated once at module load via random.Random(20260606). The template
# grammar covers ~60 system-log-style patterns x rich slot vocabularies
# (service names, status codes, host names, version strings, numbers). The
# resulting strings are deduped and trimmed to exactly 1000 unique entries.
# Each trial samples from this pool with the trial's own RNG, so trial-level
# determinism still flows from cfg.seed.
# ---------------------------------------------------------------------------

FILLER_POOL_SEED = 20260606
FILLER_POOL_SIZE = 1000

# Slot vocabularies. Wide enough that template * slots >> 1000.
_FP_SERVICES = (
    "auth", "billing", "cache", "cron", "dns", "edge", "etl", "gateway",
    "graphql", "ingest", "kafka", "ldap", "logger", "mailer", "metrics",
    "nginx", "oauth", "postgres", "queue", "redis", "router", "scheduler",
    "search", "session", "smtp", "ssh", "storage", "stream", "sync", "webhook",
)
_FP_HOSTS = (
    "alpha-01", "alpha-02", "bravo-03", "bravo-04", "charlie-05",
    "delta-06", "echo-07", "foxtrot-08", "golf-09", "hotel-10",
    "india-11", "juliet-12", "kilo-13", "lima-14", "mike-15",
    "node-a", "node-b", "node-c", "node-d", "node-e",
    "rack-12", "rack-13", "rack-14", "edge-eu", "edge-us", "edge-ap",
)
_FP_REGIONS = (
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-central-1", "ap-south-1", "ap-northeast-1",
    "sa-east-1", "ca-central-1",
)
_FP_STATUS_CODES = (
    "200", "201", "202", "204", "301", "302", "304",
    "400", "401", "403", "404", "409", "418", "422",
    "429", "500", "502", "503", "504",
)
_FP_COMPONENTS = (
    "ingestion pipeline", "background worker", "telemetry agent",
    "config loader", "feature flag service", "rate limiter",
    "task scheduler", "discovery client", "health probe",
    "credential manager", "audit logger", "metrics exporter",
)
_FP_SENSORS = (
    "thermal sensor", "humidity probe", "vibration monitor",
    "pressure gauge", "voltage rail", "fan controller",
    "ambient light sensor", "door contact", "smoke detector",
)
_FP_ROOMS = (
    "lobby", "loading dock", "server room", "cold aisle",
    "hot aisle", "break room", "operations center", "lab 2A",
    "lab 3B", "lab 4C",
)
_FP_DBS = (
    "users", "orders", "events", "sessions", "audit", "metrics",
    "telemetry", "billing", "catalog", "inventory",
)
_FP_QUEUES = (
    "ingest", "retry", "deadletter", "priority", "low", "high",
    "batch", "stream", "indexer", "notifier",
)
_FP_FILES = (
    "config.yaml", "schema.json", "secrets.env", "metrics.tsv",
    "audit.log", "snapshot.bin", "state.db", "manifest.toml",
    "checksums.sha256", "rollback.sql",
)
_FP_ENVS = ("production", "staging", "canary", "qa", "shadow")
_FP_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _filler_templates() -> list[str]:
    """Return ~60 distinct system-log-style sentence templates.

    Each template uses named slots filled from the vocabularies above plus the
    generic {n} (small int), {n3} (3-digit int), {v} (semver) and {ms} (latency).
    """
    return [
        # service status (10)
        "Service {svc} reports status {code} on host {host}.",
        "Heartbeat from {svc} received after {ms} ms.",
        "Service {svc} restarted on {host} ({env}).",
        "Service {svc} entered degraded mode.",
        "Service {svc} returned to nominal state.",
        "Failover triggered for {svc} in region {region}.",
        "Leader election completed for {svc}: new leader {host}.",
        "Service {svc} drained {n} connections before shutdown.",
        "Background worker {comp} finished cycle {n3}.",
        "Component {comp} reloaded configuration successfully.",
        # query/results (10)
        "Query against {db} returned {n3} rows in {ms} ms.",
        "Index rebuild on table {db} completed in {ms} ms.",
        "Cache hit ratio for {svc} is {n} percent.",
        "Replication lag on {db} is {n} seconds.",
        "Snapshot of {db} written to volume {host}.",
        "Migration {v} applied to {db} on {env}.",
        "Read replica {host} caught up to primary.",
        "Vacuum on {db} reclaimed {n3} pages.",
        "Slow query log recorded {n} entries.",
        "Connection pool for {svc} resized to {n} workers.",
        # config changes (8)
        "Configuration file {file} reloaded on {host}.",
        "Feature flag '{svc}_beta' set to enabled.",
        "Feature flag '{svc}_beta' set to disabled.",
        "Threshold for {comp} adjusted to {n} percent.",
        "Rotation policy updated for {file}.",
        "Secret {file} rotated by {comp}.",
        "Deployment of version {v} started on {env}.",
        "Deployment of version {v} completed on {env}.",
        # sensor readings (10)
        "{sensor} in {room} reads {n} units.",
        "{sensor} on {host} reports nominal range.",
        "{sensor} threshold exceeded in {room}.",
        "CPU temperature on {host} stable at {n}C.",
        "Fan speed on {host} set to {n} percent.",
        "Power draw on rack {host} is {n3} watts.",
        "Ambient temperature in {room} measured at {n}C.",
        "Humidity in {room} is {n} percent.",
        "Vibration on {host} within tolerance.",
        "Door contact in {room} reports closed.",
        # network/security (10)
        "Network latency to {region} is {ms} ms.",
        "Packet loss to {host} is {n} percent.",
        "Firewall rule {n3} updated on {host}.",
        "TLS certificate for {svc} renewed for {n} days.",
        "Login from internal node {host}.",
        "Audit event {n3} recorded by {comp}.",
        "Suspicious request blocked by {svc} (code {code}).",
        "Session {n3} closed by {svc}.",
        "Rate limit reached for endpoint /v1/{svc}.",
        "DNS lookup for {svc}.internal resolved in {ms} ms.",
        # storage/queue/backup (10)
        "Backup job {n3} for {db} finished successfully.",
        "Backup job {n3} for {db} encountered warning code {code}.",
        "Disk usage on {host} is {n} percent.",
        "Volume {host} mounted read-only for maintenance.",
        "Queue '{queue}' depth is {n3} messages.",
        "Queue '{queue}' drained in {ms} ms.",
        "Dead-letter queue '{queue}' has {n} stale entries.",
        "Archive of {file} uploaded to cold storage.",
        "Object {file} replicated to region {region}.",
        "Garbage collection on {svc} freed {n3} megabytes.",
        # misc ops (5)
        "Maintenance window scheduled for {day}.",
        "Routine inspection of {comp} completed without findings.",
        "Calibration of {sensor} in {room} completed.",
        "Inventory check on rack {host} marked complete.",
        "On-call rotation for {svc} handed over.",
    ]


def _build_filler_pool(seed: int = FILLER_POOL_SEED,
                       size: int = FILLER_POOL_SIZE) -> list[str]:
    """Build the deterministic 1000-string filler pool.

    Templates are filled with random combinations from the slot vocabularies
    using a fresh ``random.Random(seed)``. We oversample, dedupe by string
    identity, and slice to exactly ``size`` unique entries. Raises if the
    template/slot space cannot fill ``size`` unique strings.
    """
    rng = random.Random(seed)
    templates = _filler_templates()
    out: list[str] = []
    seen: set[str] = set()

    # Hard cap on attempts. Template count * average distinct slot products
    # easily exceeds 1e5, so this is plenty of headroom.
    max_attempts = size * 200
    attempts = 0
    while len(out) < size and attempts < max_attempts:
        attempts += 1
        tmpl = rng.choice(templates)
        s = tmpl.format(
            svc=rng.choice(_FP_SERVICES),
            host=rng.choice(_FP_HOSTS),
            region=rng.choice(_FP_REGIONS),
            code=rng.choice(_FP_STATUS_CODES),
            comp=rng.choice(_FP_COMPONENTS),
            sensor=rng.choice(_FP_SENSORS),
            room=rng.choice(_FP_ROOMS),
            db=rng.choice(_FP_DBS),
            queue=rng.choice(_FP_QUEUES),
            file=rng.choice(_FP_FILES),
            env=rng.choice(_FP_ENVS),
            day=rng.choice(_FP_DAYS),
            v=f"{rng.randint(1, 6)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}",
            n=rng.randint(1, 99),
            n3=rng.randint(100, 999),
            ms=rng.randint(1, 500),
        )
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    if len(out) < size:
        raise RuntimeError(
            f"FILLER_POOL build failed: only {len(out)}/{size} unique strings "
            f"after {attempts} attempts; widen templates or slot vocabularies."
        )
    assert len(out) == size
    assert len(set(out)) == size
    return out


FILLER_POOL: list[str] = _build_filler_pool()

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
# ---------------------------------------------------------------------------
# Trivial T1 baseline (added 2026-07-20)
# ---------------------------------------------------------------------------
# A tagged "[Packet xx - T1]" task that requires no computation: report a given
# word. Purpose: schema-preserving baseline. The 'none' load puts an UNTAGGED
# filler in the T1 slot, so 'none' differs from real loads in schema validity
# (no T1 marker, yet the template demands "Target 1 Result:"), not just in
# load. 'trivial' isolates the load factor: same stream schema, ~zero demand.
# 100 distinct words; none is a NATO alphabet word (those are reserved for the
# T2 passphrase and the mask decoy).

_TRIVIAL_WORDS = (
    "BLUE", "RED", "GREEN", "YELLOW", "PURPLE", "ORANGE", "BLACK", "WHITE",
    "BROWN", "PINK", "GRAY", "SILVER", "GOLD", "CYAN", "MAGENTA", "CRIMSON",
    "VIOLET", "INDIGO", "TEAL", "MAROON",
    "CAT", "DOG", "HORSE", "COW", "SHEEP", "GOAT", "PIG", "DUCK", "GOOSE",
    "OWL", "FOX", "WOLF", "BEAR", "DEER", "MOOSE", "OTTER", "SEAL", "WHALE",
    "SHARK", "CRAB", "FROG", "TOAD", "SNAKE", "EAGLE", "HAWK", "CROW",
    "ROBIN", "FINCH", "TROUT", "SALMON",
    "TABLE", "CHAIR", "SPOON", "FORK", "KNIFE", "PLATE", "CUP", "BOWL",
    "LAMP", "CLOCK", "DOOR", "WINDOW", "FLOOR", "ROOF", "WALL", "BRICK",
    "STONE", "RIVER", "LAKE", "OCEAN", "CLOUD", "RAIN", "SNOW", "WIND",
    "STORM", "TREE", "LEAF", "ROOT", "BRANCH", "FLOWER",
    "BREAD", "MILK", "HONEY", "SUGAR", "SALT", "PEPPER", "APPLE", "LEMON",
    "GRAPE", "PEACH", "CHERRY", "MELON", "WHEAT", "CORN", "RICE", "BEAN",
    "ONION", "CARROT", "TOMATO", "GARLIC",
)

T1_TRIVIAL_BANK = tuple(
    (f"Your Target 1 task in this stream is trivial: "
     f"report the word {w} as your Target 1 result.", w)
    for w in _TRIVIAL_WORDS
)

_VALID_LOADS = (
    "none",
    "trivial",
    *_SEMANTIC_ALIASES,
    *_MATH_ALIASES,
)


def _get_t1(load: str, rng: random.Random):
    """Returns (t1_line, t1_answer, tag).  load=='none' -> (None, None, '')."""
    if load == "none":
        return None, None, ""
    if load == "trivial":
        task, ans = rng.choice(T1_TRIVIAL_BANK)
        return f"LOGIC OVERRIDE. {task}", ans, " - T1"
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
    # Sample from the deterministic 1000-phrase FILLER_POOL (Item 7).
    # rng is the per-trial RNG so trial-level determinism is preserved.
    return rng.choice(FILLER_POOL)


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
    n_pre: int | None = None      # fillers before T1 (None = auto: fill up to total_packets)
    n_post: int | None = None     # fillers after T2 (None = random in [1, budget-1]; int = fixed)
    total_packets: int = 15       # total stream length, incl. T1, T2, mask, fillers, 'End of stream'
    t2_words: int = 3
    target_t2_abs_index: int = 12  # LEGACY, no longer used: T2 position now follows from
                                   # total_packets and n_post (index = total_packets - 1 - n_post)
    seed: int = 0
    # --- combined A×B×H design (2026-07-20) ---------------------------------
    # n_tasks=None -> legacy single-T1 design (everything above applies).
    # n_tasks=int (1..total_packets) -> task-stream design: n_tasks tasks
    # (1 passphrase + n_tasks-1 load tasks of level t1_load) scattered among
    # total_packets numbered packets; 'End of stream.' becomes an unnumbered
    # closing line, so n_tasks=total_packets means tasks-only. lag / mask /
    # n_pre / n_post / target_t2_abs_index are IGNORED in this mode.
    n_tasks: int | None = None
    naming: str = "ordered"        # "ordered" (Task 1..n, stream order) | "non-ordered"
    passphrase_last: bool = True   # passphrase = last task; False -> random rank
    # Anti-enumeration instruction (2026-07-22, idea I1): when True the cot
    # OUTPUT FORMAT explicitly forbids re-enumerating the stream packet by
    # packet inside <Thinking>. Task-design cot only; no effect on direct or
    # legacy trials, and NO effect on any RNG draw (pure prompt-text change).
    # Default True for new runs; rescore_graded defaults it to False for CSVs
    # predating the column, which reproduces the old prompt byte-exactly.
    anti_enumeration: bool = True


@dataclass
class Trial:
    system: str
    user: str                    # the full user prompt (rules + example + stream + template)
    t2_phrase: str
    t1_answer: str | None
    template_prefix_after_prompt: str  # empty-CoT answer template up to the T2 slot; used only
                                       # by the optional encoding-baseline control (see
                                       # experiment.run_trial(encoding_baseline=True))
    config: TrialConfig = field(default=None)
    n_pre_used: int = 0          # actually used n_pre (resolves None to auto value)
    n_post_used: int = 0         # actually used n_post (resolves None to the random draw)
    t2_abs_index: int = 0        # 1-based absolute packet index where T2 landed
    # --- combined-design extras (2026-07-20) ---------------------------------
    tasks: list = field(default_factory=list)  # per-task dicts: name, kind,
                                               # packet, rank, question, answer
    forced_close_text: str = ""    # injected by protocol.py when the finite
                                   # CoT budget forces the block closed
    t2_task_name: str | None = None  # name of the passphrase task (new design)
    t2_rank: int | None = None       # 1-based rank of the passphrase task


SYSTEM = "You are an automated data-extraction system."

RULES = ("You will be provided with a stream of sequential data packets. Read the entire stream, "
         "then output EXACTLY the template at the end. Do not add conversational text, "
         "introductory phrases, or extra punctuation.")


# ---------------------------------------------------------------------------
# Worked example (fixed, no RNG). Prefixed to every prompt so the model sees a
# fully-filled example of the OUTPUT TEMPLATE. Helps Qwen avoid emitting an
# empty <Thinking>...</Thinking> block (Item 1).
# ---------------------------------------------------------------------------

_WORKED_EXAMPLE_STREAM = (
    "[Packet 01]: System initialization complete. Weather is clear.\n"
    "[Packet 02]: Update 1.0.3 applied to main server.\n"
    "[Packet 03 - T1]: LOGIC OVERRIDE. Decide whether SALMON is an ANIMAL or a TOOL. "
    "Answer ANIMAL or TOOL.\n"
    "[Packet 04]: Sensor array recalibrated.\n"
    "[Packet 05]: Disk usage at 42 percent.\n"
    '[Packet 06 - T2]: The secure operational passphrase is "VICTOR XRAY ROMEO".\n'
    "[Packet 07]: Backup job 17 finished successfully.\n"
    "[Packet 08]: End of stream."
)

WORKED_EXAMPLE_COT = (
    "EXAMPLE (illustration only - not part of the real stream):\n"
    "DATA STREAM:\n"
    f"{_WORKED_EXAMPLE_STREAM}\n"
    "OUTPUT:\n"
    "<Thinking>\n"
    "The T1 packet asks whether SALMON is an ANIMAL or a TOOL; a salmon is a fish, "
    "so the answer is ANIMAL. The T2 packet contains the passphrase "
    '"VICTOR XRAY ROMEO".\n'
    "</Thinking>\n"
    "<Final_Answers>\n"
    "Target 1 Result: ANIMAL\n"
    "Target 2 Result: VICTOR XRAY ROMEO\n"
    "</Final_Answers>\n"
    "Now process the following ACTUAL stream:"
)

WORKED_EXAMPLE_DIRECT = (
    "EXAMPLE (illustration only - not part of the real stream):\n"
    "DATA STREAM:\n"
    f"{_WORKED_EXAMPLE_STREAM}\n"
    "OUTPUT:\n"
    "<Final_Answers>\n"
    "Target 2 Result: VICTOR XRAY ROMEO\n"
    "Target 1 Result: ANIMAL\n"
    "</Final_Answers>\n"
    "Now process the following ACTUAL stream:"
)


def _worked_example(regime: str) -> str:
    """Return the regime-appropriate fixed worked example block."""
    if regime == "direct":
        return WORKED_EXAMPLE_DIRECT
    return WORKED_EXAMPLE_COT


def _build_trial_legacy(cfg: TrialConfig) -> Trial:
    rng = random.Random(cfg.seed)
    t2 = random_passphrase(rng, cfg.t2_words)

    t1_line, t1_ans, t1_tag = _get_t1(cfg.t1_load, rng)
    if t1_line is None:                  # 'none' load -> neutral filler in the T1 slot
        t1_line = _filler(rng)

    # layout: [n_pre fillers][T1][mask?][lag fillers][T2][n_post fillers][End of stream]
    # Fixed budget: total_packets = n_pre + n_post + (T1 + T2 + End + mask + lag).
    # n_post is drawn at random by default (or fixed via cfg.n_post); n_pre then
    # absorbs the remainder so the stream always has exactly total_packets packets.
    # Because lag+mask eat into the budget, n_post is bounded by the lag.
    # At least one filler must open the stream (n_pre >= 1) and one must follow
    # T2 (n_post >= 1), so T1 is never packet 1 and T2 never directly precedes
    # 'End of stream'. Hence T2 = total - 1 - n_post and T1 = T2 - lag - 1 >= 2.
    fixed_blocks = 3 + (1 if cfg.mask else 0) + cfg.lag   # T1 + T2 + End + mask + lag fillers
    budget = cfg.total_packets - fixed_blocks             # fillers left for n_pre + n_post
    if budget < 2:                                        # need >= 1 on each side
        raise ValueError(
            f"lag={cfg.lag} (mask={cfg.mask}) does not fit in "
            f"total_packets={cfg.total_packets} with >=1 filler at each end; "
            f"reduce lag or raise total_packets."
        )
    if cfg.n_post is None:
        n_post = rng.randint(1, budget - 1)               # random by default
    else:
        n_post = cfg.n_post
        if not 1 <= n_post <= budget - 1:
            raise ValueError(
                f"n_post={n_post} out of range [1, {budget - 1}] for lag={cfg.lag}, "
                f"mask={cfg.mask}, total_packets={cfg.total_packets} "
                f"(>=1 filler required at each end)."
            )
    if cfg.n_pre is None:
        n_pre = budget - n_post          # keeps the stream at exactly total_packets
    else:
        n_pre = cfg.n_pre                # explicit n_pre sweep (Item 2): total may then float
        if n_pre < 1:
            raise ValueError("n_pre must be >= 1 (the stream must open with a filler).")

    packets = []
    for _ in range(n_pre):
        packets.append(_filler(rng))
    packets.append((t1_line, t1_tag))
    if cfg.mask:
        packets.append(f"Reference codes logged: {random_passphrase(rng, 3)}.")
    for _ in range(cfg.lag):
        packets.append(_filler(rng))
    packets.append((f'The secure operational passphrase is "{t2}".', " - T2"))
    for _ in range(n_post):
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

    example_block = _worked_example(cfg.regime)
    header = f"{RULES}\n{example_block}\nDATA STREAM:\n{stream}\n{template}"
    return Trial(
        system=SYSTEM,
        user=header,
        t2_phrase=t2,
        t1_answer=t1_ans,
        template_prefix_after_prompt=assistant_prefix,
        forced_close_text=_LEGACY_FORCED_CLOSE if cfg.regime == "cot" else "",
        config=cfg,
        n_pre_used=n_pre,
        n_post_used=n_post,
        t2_abs_index=n_pre + 2 + (1 if cfg.mask else 0) + cfg.lag,
    )



# ===========================================================================
# Combined A×B×H design (2026-07-20): task streams
# ===========================================================================
# B: n_tasks tasks in one stream (1 passphrase + n_tasks-1 load tasks).
# H: per-trial task names ("ordered" = Task 1..n in stream order,
#    "non-ordered" = random names from TASK_NAME_BANK), hidden cardinality,
#    free report ("- Task <NAME>: <result>", one line per task found).
# A: the finite CoT budget lives in protocol.py; Trial.forced_close_text is
#    what gets injected when the budget forces the <Thinking> block closed.

_VALID_NAMINGS = ("ordered", "non-ordered")

# 100 task names: unique, uppercase, disjoint from the NATO alphabet (reserved
# for passphrases / the mask decoy), from _TRIVIAL_WORDS (trivial-task
# answers), and from the reserved worked-example names.
TASK_NAME_BANK = (
    "WATERMELON", "PYRAMID", "LANTERN", "CACTUS", "DOLPHIN", "MARBLE",
    "TROMBONE", "GLACIER", "PUMPKIN", "SAPPHIRE", "WALRUS", "ORCHID",
    "TOBOGGAN", "CHIMNEY", "FALCON", "NUTMEG", "ORIGAMI", "PELICAN",
    "QUILT", "RASPBERRY", "SUNDIAL", "TULIP", "UMBRELLA", "VOLCANO",
    "WHEELBARROW", "YOGURT", "ZEPPELIN", "ANCHOVY", "BAGPIPE", "CARNIVAL",
    "DAFFODIL", "EGGPLANT", "FLAMINGO", "GARGOYLE", "HAMMOCK", "IGLOO",
    "JIGSAW", "KAYAK", "LOBSTER", "MANDOLIN", "NARWHAL", "OBELISK",
    "PARSNIP", "QUARRY", "RHUBARB", "SCARECROW", "TAPESTRY", "UKULELE",
    "VELVET", "WOMBAT", "XYLOPHONE", "YODEL", "ZUCCHINI", "ALMOND",
    "BONSAI", "CATAPULT", "DUMPLING", "EMERALD", "FERRET", "GONDOLA",
    "HARMONICA", "ICEBERG", "JASMINE", "KETTLE", "LILAC", "METRONOME",
    "NECTARINE", "PLATYPUS", "PISTACHIO", "QUICKSAND", "RAVIOLI",
    "SAXOPHONE", "THIMBLE", "UNICYCLE", "VINEYARD", "WALNUT", "YACHT",
    "ZIGZAG", "ABACUS", "BLIZZARD", "CENTIPEDE", "DANDELION", "ESPRESSO",
    "FIREFLY", "GAZEBO", "HEDGEHOG", "INKWELL", "JUKEBOX", "KUMQUAT",
    "LULLABY", "MONGOOSE", "NOODLE", "OPAL", "PORCUPINE", "QUIVER",
    "ROOSTER", "SNORKEL", "TANGERINE", "URCHIN", "VULTURE",
)

_RESERVED_EXAMPLE_NAMES = ("CATHEDRAL", "MARMALADE")

_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Injected after a capped <Thinking> block (see protocol.py). The task-stream
# template never lists answer slots (hidden count), so the close is bare; the
# legacy template injects its first slot so the model lands on the checklist.
FORCED_CLOSE_TASKS = "</Thinking>\n<Final_Answers>\n"
_LEGACY_FORCED_CLOSE = "</Thinking>\n<Final_Answers>\nTarget 1 Result: "

RULES_TASKS = (
    "You will be provided with a stream of sequential data packets. Some packets contain a "
    "task, tagged '[Packet xx - Task <NAME>]'. Read the entire stream, then report the "
    "result of every task in the stream. Output EXACTLY in the format described at the "
    "end. Do not add conversational text, introductory phrases, or extra punctuation."
)


def _passphrase_task_line(t2: str, n_words: int) -> str:
    num = _NUM_WORDS.get(n_words, str(n_words))
    return f'Copy-paste these {num} words: "{t2}".'


def _sample_load_items(load: str, rng: random.Random, k: int) -> list:
    """k distinct (question, answer) items of the given load level."""
    if k == 0:
        return []
    if load == "none":
        raise ValueError(
            "t1_load='none' is incompatible with n_tasks > 1: load tasks need "
            "a bank. Use 'trivial' as the near-zero-demand baseline."
        )
    if load == "trivial":
        words = rng.sample(_TRIVIAL_WORDS, k)
        return [(f"This task is trivial: report the word {w} as this task's "
                 f"result.", w) for w in words]
    if load in _SEMANTIC_ALIASES:
        bank = T1_SEMANTIC_BANKS[_SEMANTIC_ALIASES[load]]
    elif load in _MATH_ALIASES:
        bank = T1_MATH_BANKS[_MATH_ALIASES[load]]
    else:
        raise ValueError(
            f"Unknown t1_load: {load!r}. Valid values: {', '.join(_VALID_LOADS)}"
        )
    return [tuple(item) for item in rng.sample(bank, k)]


def _worked_example_tasks(naming: str, regime: str) -> str:
    """Fixed worked example for the task-stream template.

    Shows 2 tasks; live test streams draw their own n_tasks, and the reserved
    example names never appear in TASK_NAME_BANK, so nothing leaks. The
    example demonstrates the '- Task <NAME>: <result>' line format and (cot)
    counting the tasks found, without any copyable placeholder text.
    """
    n1, n2 = ("1", "2") if naming == "ordered" else _RESERVED_EXAMPLE_NAMES
    stream = (
        "[Packet 01]: System initialization complete. Weather is clear.\n"
        f"[Packet 02 - Task {n1}]: Decide whether SALMON is an ANIMAL or a "
        "TOOL. Answer ANIMAL or TOOL.\n"
        "[Packet 03]: Update 1.0.3 applied to main server.\n"
        f'[Packet 04 - Task {n2}]: Copy-paste these three words: '
        '"VICTOR XRAY ROMEO".\n'
        "[Packet 05]: Backup job 17 finished successfully.\n"
        "End of stream."
    )
    thinking = (
        "<Thinking>\n"
        f"Packet 02 is Task {n1}: a salmon is a fish, so the answer is ANIMAL. "
        f"Packet 04 is Task {n2}: I must copy the three words VICTOR XRAY "
        "ROMEO. I found 2 tasks in this stream.\n"
        "</Thinking>\n"
    )
    answers = (
        "<Final_Answers>\n"
        f"- Task {n1}: ANIMAL\n"
        f"- Task {n2}: VICTOR XRAY ROMEO\n"
        "</Final_Answers>"
    )
    body = (thinking + answers) if regime == "cot" else answers
    return ("EXAMPLE (illustration only - not part of the real stream):\n"
            "DATA STREAM:\n"
            f"{stream}\n"
            "OUTPUT:\n"
            f"{body}\n"
            "Now process the following ACTUAL stream:")


def _output_format_tasks(regime: str, anti_enumeration: bool = False) -> str:
    """Answer-format spec. Prose instructions, no fill-in-the-blank placeholder
    (CoT-skip fix 1): the only literal-looking line is the per-task line format
    itself, which the worked example shows correctly instantiated.

    anti_enumeration=True (cot only) appends an explicit prohibition on
    re-enumerating the stream packet by packet inside <Thinking> (idea I1).
    False reproduces the pre-2026-07-22 prompt byte-exactly.
    """
    if regime == "direct":
        return (
            "OUTPUT FORMAT:\n"
            "A <Final_Answers> block with EXACTLY one line per task you found "
            "in the stream, each line in the form:\n"
            "- Task <NAME>: <result>"
        )
    anti = (
        " Do NOT list or summarize the stream packet by packet: skip filler "
        "packets entirely and mention ONLY the task packets you found."
        if anti_enumeration else ""
    )
    return (
        "OUTPUT FORMAT:\n"
        "First a <Thinking> block: reason step by step in your own words - "
        "identify every task packet in the stream and solve each one. Do not "
        f"copy instruction text into it.{anti}\n"
        "Then a <Final_Answers> block with EXACTLY one line per task you "
        "found in the stream, each line in the form:\n"
        "- Task <NAME>: <result>"
    )


def _build_trial_tasks(cfg: TrialConfig) -> Trial:
    """Build a combined-design (A×B×H) task-stream trial.

    RNG draw order (fixed — rescore_graded.py replays it from the logged
    config/seed): passphrase words -> task positions -> passphrase rank (only
    if passphrase_last=False) -> task names (only if non-ordered) -> load
    items -> fillers in packet order.
    """
    n, total = cfg.n_tasks, cfg.total_packets
    if not 1 <= n <= total:
        raise ValueError(f"n_tasks={n} out of range [1, {total}]")
    if cfg.naming not in _VALID_NAMINGS:
        raise ValueError(f"naming={cfg.naming!r}; valid: {_VALID_NAMINGS}")

    rng = random.Random(cfg.seed)
    t2 = random_passphrase(rng, cfg.t2_words)
    positions = sorted(rng.sample(range(1, total + 1), n))
    pp_rank = (n - 1) if cfg.passphrase_last else rng.randrange(n)
    if cfg.naming == "non-ordered":
        names = rng.sample(TASK_NAME_BANK, n)
    else:
        names = [str(i + 1) for i in range(n)]
    load_items = _sample_load_items(cfg.t1_load, rng, n - 1)

    tasks, li = [], iter(load_items)
    for rank, (pos, name) in enumerate(zip(positions, names)):
        if rank == pp_rank:
            q, a, kind = _passphrase_task_line(t2, cfg.t2_words), t2, "passphrase"
        else:
            (q, a), kind = next(li), "load"
        tasks.append(dict(name=name, kind=kind, packet=pos, rank=rank + 1,
                          question=q, answer=a))

    by_pos = {t["packet"]: t for t in tasks}
    lines = []
    for i in range(1, total + 1):
        if i in by_pos:
            t = by_pos[i]
            lines.append(f"[Packet {i:02d} - Task {t['name']}]: {t['question']}")
        else:
            lines.append(f"[Packet {i:02d}]: {_filler(rng)}")
    lines.append("End of stream.")   # unnumbered: does not consume a packet slot
    stream = "\n".join(lines)

    header = (f"{RULES_TASKS}\n{_worked_example_tasks(cfg.naming, cfg.regime)}\n"
              f"DATA STREAM:\n{stream}\n"
              f"{_output_format_tasks(cfg.regime, cfg.anti_enumeration)}")
    pp = tasks[pp_rank]
    return Trial(
        system=SYSTEM,
        user=header,
        t2_phrase=t2,
        t1_answer=None,                    # per-task answers live in .tasks
        template_prefix_after_prompt="",   # encoding-baseline control is legacy-only
        forced_close_text=FORCED_CLOSE_TASKS if cfg.regime == "cot" else "",
        config=cfg,
        n_pre_used=None,
        n_post_used=None,
        t2_abs_index=pp["packet"],
        tasks=tasks,
        t2_task_name=pp["name"],
        t2_rank=pp_rank + 1,
    )


def build_trial(cfg: TrialConfig) -> Trial:
    """Dispatch on cfg.n_tasks.

    n_tasks=None -> legacy single-T1 design (BIT-EXACT, do not touch: '
    'rescore_graded.py rebuilds old CSVs through this path).
    n_tasks=int  -> combined A×B×H task-stream design.
    """
    if cfg.n_tasks is None:
        return _build_trial_legacy(cfg)
    return _build_trial_tasks(cfg)


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

    # (6) trivial baseline pool (2026-07-20): 100 items, no dups, (str, str),
    # answer appears in its instruction, and no NATO word (reserved for T2/mask).
    nato = set(NATO)
    assert len(T1_TRIVIAL_BANK) == expected_size, (
        f"trivial: expected {expected_size} items, got {len(T1_TRIVIAL_BANK)}"
    )
    t_qs = [q for q, _ in T1_TRIVIAL_BANK]
    assert len(set(t_qs)) == len(t_qs), "trivial: duplicate instructions"
    for i, (q, a) in enumerate(T1_TRIVIAL_BANK):
        assert isinstance(q, str) and isinstance(a, str), (
            f"trivial[{i}] is not a (str, str) tuple")
        assert a in q, f"trivial[{i}]: answer {a!r} not in instruction"
        assert a not in nato, f"trivial[{i}]: answer {a!r} is a NATO word"
    if verbose:
        print("  trivial: OK (100 items, no dups, no NATO words)")

    # (7) task-name bank (2026-07-20, combined design): 100 unique uppercase
    # names, disjoint from NATO, trivial-task answers and reserved example names.
    assert len(TASK_NAME_BANK) == expected_size, (
        f"task names: expected {expected_size}, got {len(TASK_NAME_BANK)}")
    assert len(set(TASK_NAME_BANK)) == len(TASK_NAME_BANK), (
        "task names: duplicates")
    _bad = [w for w in TASK_NAME_BANK
            if (not w.isupper()) or (w in nato) or (w in set(_TRIVIAL_WORDS))
            or (w in _RESERVED_EXAMPLE_NAMES)]
    assert not _bad, f"task names invalid or colliding: {_bad[:5]}"
    if verbose:
        print("  task-name bank: OK (100 names, no collisions)")
        print("All 11 T1 pools + task-name bank validated.")


def _maybe_validate() -> None:
    import os
    if os.environ.get("LLM_BLINK_VALIDATE") == "1":
        _validate_t1_pools(verbose=True)


_maybe_validate()


if __name__ == "__main__":
    _validate_t1_pools(verbose=True)
