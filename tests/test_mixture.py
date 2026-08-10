"""Synthetic validation of the all-or-none mixture analysis (issue #14, Step 3).

This is the function that decides the paper's conclusion, so it MUST be shown
not to prefer either answer. Three generators of known type:

  * pure pi-shift (ignition): fixed betas, mixing weight moves with condition
    -> expect Mpi ~ Mfull, and Mmu strictly worse than Mfull.
  * pure mu-shift (resource sharing): fixed weight, component means move
    -> expect Mmu ~ Mfull, and Mpi strictly worse than Mfull.
  * single rate (graded, no modes): one beta per condition
    -> expect M0 competitive with Mfull (Mfull does not beat M0).

Run directly (`python3 -B tests/test_mixture.py`) for a verbose report, or via
pytest (the `test_*` functions assert the orderings).
"""
from __future__ import annotations
import os
import sys
import numpy as np

# repo dir is itself the package "LLM_Blink"; put its PARENT on the path.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_REPO))
from LLM_Blink.mixture import (  # noqa: E402
    betabinom_mixture_cv, verdict_from_cv, fit_betabinom_mixture,
    overdispersion_z, pi_shift_null, _ab_from_mu_rho, access_level, icc_nested,
)

K = 20
N_PER = 220
CONDS = ["c0", "c1", "c2"]


def _draw(pi_by_cond, mu_lo_by_cond, mu_hi_by_cond, rho_lo, rho_hi, seed):
    """Two-component beta-binomial generator; per-condition pi and means."""
    rng = np.random.default_rng(seed)
    s, cond = [], []
    for c, name in enumerate(CONDS):
        a_lo, b_lo = _ab_from_mu_rho(mu_lo_by_cond[c], rho_lo)
        a_hi, b_hi = _ab_from_mu_rho(mu_hi_by_cond[c], rho_hi)
        hi = rng.random(N_PER) < pi_by_cond[c]
        p = np.where(hi, rng.beta(a_hi, b_hi, N_PER), rng.beta(a_lo, b_lo, N_PER))
        s.append(rng.binomial(K, p))
        cond += [name] * N_PER
    return np.concatenate(s), np.full(len(cond), K), cond


def gen_pure_pi(seed=1):
    # modes FIXED (0.12 / 0.88); only the weight on the high mode moves
    return _draw(pi_by_cond=[0.8, 0.5, 0.2],
                 mu_lo_by_cond=[0.12, 0.12, 0.12],
                 mu_hi_by_cond=[0.88, 0.88, 0.88],
                 rho_lo=0.06, rho_hi=0.06, seed=seed)


def gen_pure_mu(seed=2):
    # weight FIXED (0.5); both component means slide down with condition
    return _draw(pi_by_cond=[0.5, 0.5, 0.5],
                 mu_lo_by_cond=[0.30, 0.20, 0.10],
                 mu_hi_by_cond=[0.92, 0.75, 0.58],
                 rho_lo=0.05, rho_hi=0.05, seed=seed)


def gen_single_rate(seed=3):
    # ONE graded component per condition (mean shifts), no second mode
    rng = np.random.default_rng(seed)
    s, cond = [], []
    means = [0.65, 0.5, 0.35]
    for c, name in enumerate(CONDS):
        a, b = _ab_from_mu_rho(means[c], 0.08)
        p = rng.beta(a, b, N_PER)
        s.append(rng.binomial(K, p))
        cond += [name] * N_PER
    return np.concatenate(s), np.full(len(cond), K), cond


def _run(gen, label):
    s, k, cond = gen()
    cv = betabinom_mixture_cv(s, k, cond, folds=5, seeds=2,
                              n_restarts=3, boot=800, seed=0)
    v = verdict_from_cv(cv)
    z = overdispersion_z(s, k, cond)
    print(f"\n===== {label} =====")
    print("  mean held-out logdens:",
          {m: round(x, 4) for m, x in cv["mean"].items()})
    for key, d in cv["pairs"].items():
        star = "*" if d["excludes_zero"] else " "
        print(f"    {key:>12}: {d['mean']:+.4f}  CI[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] {star}")
    print("  Tarone Z:", round(z["Z"], 2))
    print("  VERDICT:", v["call"])
    for m in ("Mfull", "Mpi", "Mmu"):
        f = cv["fits"][m]
        print(f"    {m}: mu={np.round(f.mu,2).tolist()} pi={np.round(f.pi,2).tolist()}")
    return cv, v


def test_pure_pi_reads_as_ignition():
    cv, v = _run(gen_pure_pi, "PURE PI-SHIFT (ignition)")
    assert cv["mean"]["Mpi"] >= cv["mean"]["Mmu"] - 1e-3, \
        "Mpi should predict at least as well as Mmu under a pi-shift"
    assert cv["mean"]["Mfull"] >= cv["mean"]["M0"] - 1e-3, "should be bimodal"
    assert v["pi_like"] and not v["mu_like"], f"expected pi-like, got {v['call']}"


def test_pure_mu_reads_as_graded():
    cv, v = _run(gen_pure_mu, "PURE MU-SHIFT (resource sharing)")
    assert cv["mean"]["Mmu"] >= cv["mean"]["Mpi"] - 1e-3, \
        "Mmu should predict at least as well as Mpi under a mu-shift"
    assert v["mu_like"] and not v["pi_like"], f"expected mu-like, got {v['call']}"


def test_single_rate_reads_as_M0():
    cv, v = _run(gen_single_rate, "SINGLE RATE (graded, no modes)")
    # Mfull must NOT decisively beat M0 (no genuine second mode present)
    key = "M0-Mfull"
    assert not cv["pairs"][key]["excludes_zero"] or cv["pairs"][key]["mean"] > 0, \
        "single-rate data should not read as strongly bimodal"


def test_overdispersion_contrast():
    _, _, _ = gen_single_rate()
    z_mix = overdispersion_z(*gen_pure_pi()[:2], gen_pure_pi()[2])
    # a pi-mixture is strongly overdispersed
    assert z_mix["Z"] > 3, f"mixture should be overdispersed, Z={z_mix['Z']}"


def test_access_level_and_icc():
    lvl = access_level([0.9, 0.1, 0.1], [0.9, 0.9, 0.1])
    assert lvl.tolist() == [1.0, 2.0, 3.0]
    # deterministic-within-CoT -> ICC near 1
    ids = [0, 0, 0, 1, 1, 1]
    y = [1, 1, 1, 0, 0, 0]
    assert icc_nested(ids, y) > 0.9


if __name__ == "__main__":
    import time
    t0 = time.time()
    _run(gen_pure_pi, "PURE PI-SHIFT (ignition)")
    _run(gen_pure_mu, "PURE MU-SHIFT (resource sharing)")
    _run(gen_single_rate, "SINGLE RATE (graded, no modes)")
    # pi->mu calibration band demo on the pure-pi fit
    s, k, cond = gen_pure_pi()
    fit = fit_betabinom_mixture(s, k, cond, model="Mfull", n_restarts=6)
    band = pi_shift_null(fit, K, N_PER, cond_pair=(0, -1), n_rep=200)
    print("\npi->mu null band (spurious dmu under pure-pi):",
          np.round(band["band"], 3),
          "| observed dmu_hi:",
          round(float(fit.mu[1][-1] - fit.mu[1][0]), 3))
    print(f"\n[done in {time.time()-t0:.0f}s]")
