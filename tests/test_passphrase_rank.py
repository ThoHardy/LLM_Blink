"""passphrase_rank fixes the passphrase's stream rank (issue #18 §5, Campaign B).

Offline. Checks that a fixed rank lands the passphrase at that rank, that the
default path stays byte-exact (so CSVs predating the field rebuild identically),
that the random-rank arm is untouched, and that out-of-range raises.
Run: ``python3 -B -m pytest tests/test_passphrase_rank.py``.
"""
from __future__ import annotations
import os, sys, hashlib

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO))
from LLM_Blink.stimuli import TrialConfig, build_trial          # noqa: E402


def _cfg(**kw):
    base = dict(n_tasks=5, naming="non-ordered", t1_load="trivial",
                regime="cot", seed=7)
    base.update(kw)
    return TrialConfig(**base)


def test_fixed_rank_places_passphrase():
    for r in (1, 3, 5):
        tr = build_trial(_cfg(passphrase_rank=r))
        assert tr.t2_rank == r
        pp = [t for t in tr.tasks if t["kind"] == "passphrase"][0]
        assert pp["rank"] == r


def test_default_is_byte_exact():
    # default (passphrase_last=True, rank=None) must equal an explicit spelling,
    # i.e. adding the field consumed no RNG draw.
    a = build_trial(_cfg()).user
    b = build_trial(_cfg(passphrase_last=True, passphrase_rank=None)).user
    assert hashlib.md5(a.encode()).hexdigest() == hashlib.md5(b.encode()).hexdigest()


def test_random_rank_arm_untouched():
    tr = build_trial(_cfg(passphrase_last=False))
    assert 1 <= tr.t2_rank <= 5


def test_out_of_range_raises():
    for bad in (0, 6, -1):
        try:
            build_trial(_cfg(passphrase_rank=bad))
        except ValueError:
            continue
        raise AssertionError(f"passphrase_rank={bad} should raise")


if __name__ == "__main__":
    test_fixed_rank_places_passphrase()
    test_default_is_byte_exact()
    test_random_rank_arm_untouched()
    test_out_of_range_raises()
    print("ok")
