"""All-or-none vs graded access: beta-binomial mixture on forked-resampling counts.

Issue #14, Step 3 — the analysis that decides the paper's conclusion. It does
NOT touch a GPU or any model: it operates purely on the per-trial success
counts ``(s_i, k_i)`` produced by the forked report/access probes (``resample.py``,
Steps 1-2), plus a per-trial condition label.

The scientific question (Sergent & Dehaene 2004, ported): when the serial stage
(CoT) is loaded and a co-present item fails to reach report, is the failure
**all-or-none** (ignition / global workspace — the mixing proportion pi moves
with load, the mode locations mu stay put) or **graded** (resource sharing — mu
slides with load)? This is a claim about the *shape of a distribution*, not an
effect size, so we fit the observation model INSIDE the analysis:

    a two-component beta-binomial mixture directly on the counts (s_i, k).

The binomial layer absorbs the k=20 sampling noise; the Beta components live on
the latent per-trial rate; pi and mu keep their meanings. Each Beta component is
parameterised by its mean ``mu`` and overdispersion ``rho`` in (0, 1) rather
than (a, b): interpretable, and tying across conditions is trivial. rho -> 0 is
a spike at mu (binomial); rho -> 1 pushes mass to {0, k}.

Competing models (fitted by EM; §4.1 of the ticket):

    M0    one component,  mu free per condition                 -> pure graded, no modes
    Mpi   two components, mu & rho tied, pi free per condition  -> ignition
    Mmu   two components, pi & rho tied, mu free per condition  -> resource sharing
    Mfull two components, pi & mu free per condition            -> unconstrained ref

rho is tied across conditions in every two-component model (the modes are stable
objects; only their *weight* or *location* is allowed to move, per the model).

Model comparison is by **held-out predictive log-likelihood** with a paired
bootstrap over trials — NOT BIC/AIC (no asymptotic penalty has a claim on a
bounded mixture quantity at n~200/cell) and NOT a t-test across folds (fold
scores are correlated; Bengio & Grandvalet 2004). See ``betabinom_mixture_cv``.

Predictions:
  all-or-none : CV(Mpi) ~ CV(Mfull)  and  CV(Mmu) < CV(Mfull); both beat M0.
  graded      : the mirror image, with M0 competitive.

Everything here is validated on synthetic generators of known type in
``tests/test_mixture.py`` — the function must be shown NOT to prefer either
answer before it is trusted on real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# Numerical guards: keep mu, rho strictly inside (0, 1).
_EPS = 1e-6
_MU_LO, _MU_HI = _EPS, 1.0 - _EPS
_RHO_LO, _RHO_HI = 1e-4, 1.0 - 1e-4


# ---------------------------------------------------------------------------
# beta-binomial in (mean, overdispersion) parameterisation
# ---------------------------------------------------------------------------

def _ab_from_mu_rho(mu: float, rho: float):
    """(mu in (0,1), rho in (0,1)) -> Beta(a, b).

    mu = a / (a + b) is the component mean; rho = 1 / (a + b + 1) is the
    intra-class correlation (the beta-binomial overdispersion). rho -> 0 gives
    a -> inf (Beta collapses to a spike at mu, i.e. a plain Binomial); rho -> 1
    gives a, b -> 0 (Beta piles mass at 0 and 1).
    """
    mu = np.clip(mu, _MU_LO, _MU_HI)
    rho = np.clip(rho, _RHO_LO, _RHO_HI)
    nu = 1.0 / rho - 1.0          # = a + b
    a = mu * nu
    b = (1.0 - mu) * nu
    return a, b


def _betaln(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def bb_logpmf(s, k, mu, rho):
    """log P(s of k) under Beta-Binomial(mu, rho). Vectorised in s, k, mu, rho.

    Manual gammaln implementation (scipy.stats.betabinom.logpmf is ~50x slower
    in the EM hot loop). BB(s;k,a,b) = logC(k,s) + betaln(s+a,k-s+b) - betaln(a,b).
    """
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    a, b = _ab_from_mu_rho(mu, rho)
    logbinom = gammaln(k + 1) - gammaln(s + 1) - gammaln(k - s + 1)
    return logbinom + _betaln(s + a, k - s + b) - _betaln(a, b)


def _bb_ll_pre(s, k, a, b, logbinom):
    """Beta-binomial log-pmf with the (s,k)-only binomial term precomputed."""
    return logbinom + _betaln(s + a, k - s + b) - _betaln(a, b)


# logit transforms so the optimiser works unconstrained
def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# fitted-model container
# ---------------------------------------------------------------------------

@dataclass
class MixtureFit:
    """A fitted (1- or 2-component) beta-binomial mixture over C conditions.

    mu:  (n_comp, C) component means per condition (tied models repeat columns).
    rho: (n_comp,)   component overdispersions (tied across conditions).
    pi:  (C,)        weight on the HIGH component per condition (all ones for
                     the 1-component model M0).
    Components are sorted so row 0 is the LOW-mean mode and row 1 the HIGH.
    """
    name: str
    n_comp: int
    mu: np.ndarray
    rho: np.ndarray
    pi: np.ndarray
    loglik: float
    n_params: int
    conditions: list = field(default_factory=list)   # condition labels, index-aligned

    def logdensity(self, s, k, cond_idx) -> np.ndarray:
        """Per-trial log marginal density under this fit (held-out scoring)."""
        s = np.asarray(s); k = np.asarray(k); cond_idx = np.asarray(cond_idx)
        if self.n_comp == 1:
            return bb_logpmf(s, k, _col(self.mu[0], cond_idx), self.rho[0])
        lo = np.log(np.clip(1 - self.pi[cond_idx], 1e-12, 1)) + \
            bb_logpmf(s, k, _col(self.mu[0], cond_idx), self.rho[0])
        hi = np.log(np.clip(self.pi[cond_idx], 1e-12, 1)) + \
            bb_logpmf(s, k, _col(self.mu[1], cond_idx), self.rho[1])
        return np.logaddexp(lo, hi)


def _col(mu_row: np.ndarray, cond_idx: np.ndarray) -> np.ndarray:
    """Gather per-trial means from a per-condition mean row."""
    return np.asarray(mu_row)[cond_idx]


# ---------------------------------------------------------------------------
# EM fit
# ---------------------------------------------------------------------------

_MODEL_SPECS = {
    # (n_comp, mu_per_cond, pi_per_cond)
    "M0":    (1, True,  False),
    "Mpi":   (2, False, True),
    "Mmu":   (2, True,  False),
    "Mfull": (2, True,  True),
}


def _n_free_params(spec, C):
    n_comp, mu_per_cond, pi_per_cond = spec
    if n_comp == 1:
        return C + 1                       # mu[c] + rho
    n_mu = (2 * C) if mu_per_cond else 2
    n_pi = C if pi_per_cond else 1
    n_rho = 2
    return n_mu + n_pi + n_rho


def _mstep_component(s, k, resp, cond_idx, C, mu_per_cond,
                     mu0, rho0, logbinom):
    """Maximise the responsibility-weighted BB log-likelihood for ONE component.

    Returns (mu_vec length C, rho scalar). rho is shared across conditions; mu
    is per-condition if mu_per_cond else a single value broadcast to all C.
    Fully vectorised: per-trial (a, b) are gathered from the per-condition mu.
    """
    w = resp
    if w.sum() < 1e-8:                       # empty component: leave as-is
        return np.full(C, np.mean(mu0)), rho0

    n_mu = C if mu_per_cond else 1

    def neg_wll(theta):
        mus = _sigmoid(theta[:n_mu])
        rho = _sigmoid(theta[n_mu])
        mu_t = mus[cond_idx] if mu_per_cond else mus[0]
        a, b = _ab_from_mu_rho(mu_t, rho)
        lp = _bb_ll_pre(s, k, a, b, logbinom)
        return -np.dot(w, lp)

    mu_init = np.atleast_1d(mu0).astype(float)
    if mu_per_cond and mu_init.size == 1:
        mu_init = np.full(C, mu_init[0])
    theta0 = np.concatenate([_logit(np.clip(mu_init, _MU_LO, _MU_HI)),
                             [_logit(np.clip(rho0, _RHO_LO, _RHO_HI))]])
    res = minimize(neg_wll, theta0, method="L-BFGS-B")
    theta = res.x
    mus = _sigmoid(theta[:n_mu])
    rho = float(np.clip(_sigmoid(theta[n_mu]), _RHO_LO, _RHO_HI))
    mu_vec = mus if mu_per_cond else np.full(C, mus[0])
    return mu_vec, rho


def _fit_once(s, k, cond_idx, C, spec, rng, logbinom, max_iter=300, tol=1e-6):
    n_comp, mu_per_cond, pi_per_cond = spec
    N = len(s)

    def _lp(mu_row, rho_scalar):
        a, b = _ab_from_mu_rho(mu_row[cond_idx], rho_scalar)
        return _bb_ll_pre(s, k, a, b, logbinom)

    if n_comp == 1:
        # single component: one EM "M-step" with all-ones responsibility.
        mu_vec, rho = _mstep_component(
            s, k, np.ones(N), cond_idx, C, mu_per_cond,
            mu0=np.clip((s / np.maximum(k, 1)).mean(), 0.05, 0.95), rho0=0.1,
            logbinom=logbinom)
        mu = mu_vec[None, :]
        rho_arr = np.array([rho])
        pi = np.ones(C)
        ll = float(_lp(mu[0], rho).sum())
        return mu, rho_arr, pi, ll

    # ---- two-component EM -------------------------------------------------
    base = (s / np.maximum(k, 1))
    mu_lo = np.clip(np.full(C, 0.10 + 0.10 * rng.random()), 0.02, 0.4)
    mu_hi = np.clip(np.full(C, 0.90 - 0.10 * rng.random()), 0.6, 0.98)
    rho = np.array([0.05 + 0.1 * rng.random(), 0.05 + 0.1 * rng.random()])
    pi = np.clip(np.full(C, base.mean() if base.size else 0.5)
                 + 0.05 * (rng.random(C) - 0.5), 0.05, 0.95)
    mu = np.vstack([mu_lo, mu_hi])

    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step: responsibility of the HIGH component per trial
        log_lo = np.log(np.clip(1 - pi[cond_idx], 1e-12, 1)) + _lp(mu[0], rho[0])
        log_hi = np.log(np.clip(pi[cond_idx], 1e-12, 1)) + _lp(mu[1], rho[1])
        denom = np.logaddexp(log_lo, log_hi)
        ll = float(denom.sum())
        r_hi = np.exp(log_hi - denom)
        r_lo = 1.0 - r_hi

        # M-step: pi (closed form), then mu/rho per component (numerical)
        if pi_per_cond:
            for c in range(C):
                m = cond_idx == c
                pi[c] = np.clip(r_hi[m].mean() if m.any() else pi[c], 1e-3, 1 - 1e-3)
        else:
            pi[:] = np.clip(r_hi.mean(), 1e-3, 1 - 1e-3)

        mu_lo_vec, rho_lo = _mstep_component(
            s, k, r_lo, cond_idx, C, mu_per_cond, mu0=mu[0], rho0=rho[0],
            logbinom=logbinom)
        mu_hi_vec, rho_hi = _mstep_component(
            s, k, r_hi, cond_idx, C, mu_per_cond, mu0=mu[1], rho0=rho[1],
            logbinom=logbinom)
        mu = np.vstack([mu_lo_vec, mu_hi_vec])
        rho = np.array([rho_lo, rho_hi])

        if ll - prev_ll < tol:
            break
        prev_ll = ll

    return mu, rho, pi, ll


def fit_betabinom_mixture(s, k, cond, model: str = "Mfull",
                          n_restarts: int = 6, seed: int = 0) -> MixtureFit:
    """Fit one of M0 / Mpi / Mmu / Mfull to counts (s, k) with condition labels.

    s, k : per-trial success counts and sample sizes (array-like, same length).
    cond : per-trial condition label (any hashable; e.g. load level or rank).
    Returns the best-of-``n_restarts`` :class:`MixtureFit` (highest log-lik).
    """
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    conds = list(dict.fromkeys(cond))          # stable unique
    cidx = {c: i for i, c in enumerate(conds)}
    cond_idx = np.array([cidx[c] for c in cond])
    C = len(conds)
    spec = _MODEL_SPECS[model]
    logbinom = gammaln(k + 1) - gammaln(s + 1) - gammaln(k - s + 1)

    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        mu, rho, pi, ll = _fit_once(s, k, cond_idx, C, spec, rng, logbinom)
        if (best is None) or (ll > best[3]):
            best = (mu, rho, pi, ll)

    mu, rho, pi, ll = best
    # sort components so row 0 is the low mode (by mean over conditions)
    if spec[0] == 2 and mu[0].mean() > mu[1].mean():
        mu = mu[::-1].copy()
        rho = rho[::-1].copy()
        pi = 1.0 - pi
    return MixtureFit(name=model, n_comp=spec[0], mu=mu, rho=rho, pi=pi,
                      loglik=ll, n_params=_n_free_params(spec, C),
                      conditions=conds)


# ---------------------------------------------------------------------------
# cross-validated model comparison (§4.1)
# ---------------------------------------------------------------------------

def _stratified_folds(cond_idx, strat, folds, rng):
    """Assign each trial to a fold, balanced within each (condition, strat) group."""
    assign = np.full(len(cond_idx), -1, dtype=int)
    groups = {}
    for i, (c, st) in enumerate(zip(cond_idx, strat)):
        groups.setdefault((c, st), []).append(i)
    for _, idxs in groups.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            assign[i] = j % folds
    return assign


def betabinom_mixture_cv(s, k, cond, strat=None,
                         models=("M0", "Mpi", "Mmu", "Mfull"),
                         folds: int = 10, seeds: int = 5,
                         n_restarts: int = 4, boot: int = 2000, seed: int = 0):
    """Stratified K-fold CV of the mixture models, paired-bootstrap comparison.

    Returns a dict with, per model, the per-trial held-out log-densities
    (averaged over fold seeds), and, per model pair, the paired-bootstrap CI on
    the mean per-trial difference. Also returns the full-data refit of each
    model (for pi/mu reporting).

    strat : optional per-trial extra stratifier (e.g. report_correct). Folds are
            balanced within each (condition x strat) group so every fold carries
            the same condition balance (§4.1 step 2).
    boot  : paired-bootstrap replicates over trials (§4.1 step 4).
    """
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    cond = list(cond)
    conds = list(dict.fromkeys(cond))
    cidx = {c: i for i, c in enumerate(conds)}
    cond_idx = np.array([cidx[c] for c in cond])
    N = len(s)
    if strat is None:
        strat = np.zeros(N, dtype=int)
    strat = np.asarray(strat)

    # per-trial held-out log density, accumulated over fold seeds
    scores = {m: np.zeros(N) for m in models}
    counts = np.zeros(N)
    for fs in range(seeds):
        rng = np.random.default_rng(seed * 97 + fs)
        assign = _stratified_folds(cond_idx, strat, folds, rng)
        for f in range(folds):
            te = assign == f
            trn = ~te
            if not te.any() or not trn.any():
                continue
            for m in models:
                fit = fit_betabinom_mixture(
                    s[trn], k[trn], np.asarray(cond)[trn],
                    model=m, n_restarts=n_restarts, seed=seed * 7 + fs)
                # map test conds through the SAME condition order as the fit
                tcidx = np.array([fit.conditions.index(c)
                                  for c in np.asarray(cond)[te]])
                scores[m][te] += fit.logdensity(s[te], k[te], tcidx)
            counts[te] += 1
    for m in models:
        scores[m] = scores[m] / np.maximum(counts, 1)

    # paired bootstrap over trials on mean per-trial differences
    pairs = {}
    rng = np.random.default_rng(seed * 131 + 1)
    idx_all = np.arange(N)
    for a in models:
        for b in models:
            if a >= b:
                continue
            diff = scores[a] - scores[b]
            boot_means = np.empty(boot)
            for t in range(boot):
                bi = rng.integers(0, N, N)
                boot_means[t] = diff[bi].mean()
            lo, hi = np.percentile(boot_means, [2.5, 97.5])
            pairs[f"{a}-{b}"] = dict(mean=float(diff.mean()),
                                     ci=(float(lo), float(hi)),
                                     excludes_zero=bool(lo > 0 or hi < 0))

    fits = {m: fit_betabinom_mixture(s, k, cond, model=m,
                                     n_restarts=max(n_restarts, 6), seed=seed)
            for m in models}
    mean_scores = {m: float(scores[m].mean()) for m in models}
    return dict(per_trial=scores, mean=mean_scores, pairs=pairs,
                fits=fits, conditions=conds, n=N)


def verdict_from_cv(cv) -> dict:
    """Turn a betabinom_mixture_cv result into the §4.1 all-or-none/graded call.

    all-or-none : Mpi ~ Mfull (CI straddles 0)  AND  Mmu < Mfull (CI excludes 0)
    graded      : Mmu ~ Mfull                    AND  Mpi < Mfull (CI excludes 0),
                  M0 competitive.
    Also requires Mfull > M0 as the bimodality precondition.
    """
    p = cv["pairs"]

    def better(a, b):   # is a strictly better than b (CI on a-b excludes 0, mean>0)
        key = f"{a}-{b}" if f"{a}-{b}" in p else f"{b}-{a}"
        d = p[key]
        m = d["mean"] if key == f"{a}-{b}" else -d["mean"]
        lo, hi = d["ci"] if key == f"{a}-{b}" else (-d["ci"][1], -d["ci"][0])
        return d["excludes_zero"] and m > 0

    def tied(a, b):
        key = f"{a}-{b}" if f"{a}-{b}" in p else f"{b}-{a}"
        return not p[key]["excludes_zero"]

    bimodal = better("Mfull", "M0") or better("Mpi", "M0") or better("Mmu", "M0")
    pi_like = tied("Mpi", "Mfull") and better("Mfull", "Mmu")
    mu_like = tied("Mmu", "Mfull") and better("Mfull", "Mpi")

    if not bimodal:
        call = "graded (no bimodality: Mfull does not beat M0)"
    elif pi_like and not mu_like:
        call = "all-or-none (pi-shift: Mpi~Mfull, Mmu<Mfull)"
    elif mu_like and not pi_like:
        call = "graded (mu-shift: Mmu~Mfull, Mpi<Mfull)"
    else:
        call = "ambiguous"
    return dict(call=call, bimodal=bimodal, pi_like=pi_like, mu_like=mu_like,
                mean_scores=cv["mean"])


# ---------------------------------------------------------------------------
# calibrated pi->mu null (§4.2.3): parametric bootstrap band for spurious dmu
# ---------------------------------------------------------------------------

def pi_shift_null(fit: MixtureFit, k: int, n_per_cond, cond_pair=(0, -1),
                  n_rep: int = 1000, seed: int = 0):
    """Parametric-bootstrap band for spurious Delta-mu under a PURE pi-shift.

    Simulates data from a pure-ignition generator matched to ``fit`` (its rho,
    its component means AVERAGED across conditions so mu is truly fixed, and its
    per-condition pi), refits Mfull, and records the distribution of
    ``mu_hi[cond_b] - mu_hi[cond_a]`` — the drift a pi-only world produces
    through the estimator alone. Observed Delta-mu counts as evidence for
    gradedness only if it exceeds this band (§4.2).

    Returns dict(band=(2.5%,97.5%), draws=array, mu_fixed=(lo,hi)).
    """
    if fit.n_comp != 2:
        raise ValueError("pi_shift_null needs a two-component fit (Mfull).")
    ca, cb = cond_pair
    C = fit.mu.shape[1]
    ca = ca % C; cb = cb % C
    mu_lo = float(fit.mu[0].mean())          # FIXED means (pure pi-shift world)
    mu_hi = float(fit.mu[1].mean())
    rho = fit.rho.copy()
    if np.ndim(n_per_cond) == 0:
        n_per_cond = [int(n_per_cond)] * C
    rng = np.random.default_rng(seed)

    a_lo, b_lo = _ab_from_mu_rho(mu_lo, rho[0])
    a_hi, b_hi = _ab_from_mu_rho(mu_hi, rho[1])
    draws = np.empty(n_rep)
    for r in range(n_rep):
        s_all, cond_all = [], []
        for c in range(C):
            n = n_per_cond[c]
            hi_mask = rng.random(n) < fit.pi[c]
            p = np.where(hi_mask,
                         rng.beta(a_hi, b_hi, n),
                         rng.beta(a_lo, b_lo, n))
            s_all.append(rng.binomial(k, p))
            cond_all.extend([c] * n)
        s_all = np.concatenate(s_all)
        try:
            rf = fit_betabinom_mixture(s_all, np.full(len(s_all), k),
                                       cond_all, model="Mfull",
                                       n_restarts=3, seed=r)
            draws[r] = rf.mu[1][cb] - rf.mu[1][ca]
        except Exception:
            draws[r] = np.nan
    draws = draws[~np.isnan(draws)]
    band = tuple(np.percentile(draws, [2.5, 97.5])) if draws.size else (np.nan, np.nan)
    return dict(band=band, draws=draws, mu_fixed=(mu_lo, mu_hi))


# ---------------------------------------------------------------------------
# Tarone's Z: overdispersion of counts vs a single common rate per cell (§4.1)
# ---------------------------------------------------------------------------

def overdispersion_z(s, k, cond=None):
    """Tarone's Z test of extra-binomial variation against one rate per condition.

    A single graded rate per cell is barely overdispersed (Z ~ 0); a genuine
    mixture is strongly overdispersed (Z large positive). Reported ALONGSIDE the
    CV comparison, never instead of it. If ``cond`` is given, Z is pooled across
    conditions after centring each on its own rate. Returns dict(Z, p_onesided).
    """
    from scipy.stats import norm
    s = np.asarray(s, dtype=float); k = np.asarray(k, dtype=float)
    if cond is None:
        cond = np.zeros(len(s), dtype=int)
    cond = np.asarray(cond)
    num = 0.0
    den = 0.0
    for c in np.unique(cond):
        m = cond == c
        sc, kc = s[m], k[m]
        if kc.sum() == 0:
            continue
        phat = sc.sum() / kc.sum()
        if phat <= 0 or phat >= 1:
            continue
        num += ((sc - kc * phat) ** 2 / (phat * (1 - phat))).sum() - kc.sum()
        den += (kc * (kc - 1)).sum()
    if den <= 0:
        return dict(Z=float("nan"), p_onesided=float("nan"))
    Z = num / np.sqrt(2.0 * den)
    return dict(Z=float(Z), p_onesided=float(norm.sf(Z)))


# ---------------------------------------------------------------------------
# access taxonomy (§5) and nested ICC (§4.3)
# ---------------------------------------------------------------------------

def access_level(report_rate, access_rate, hi: float = 0.5, lo: float = 0.5):
    """3-level access taxonomy (§5), thresholds as arguments.

    1 = conscious access & report (reported high) ; 2 = accessed, not reported
    (report low, access high) ; 3 = not accessed (both low). Returns an int
    array; np.nan where access_rate is undefined (direct regime).
    """
    rr = np.asarray(report_rate, dtype=float)
    ar = np.asarray(access_rate, dtype=float)
    out = np.full(len(rr), np.nan)
    reported = rr >= hi
    accessed = ar >= lo
    out[reported] = 1
    out[(~reported) & accessed] = 2
    out[(~reported) & (~accessed)] = 3
    return out


def icc_nested(cot_ids, report_success):
    """Intraclass correlation of report outcomes within a CoT (nested probe, §4.3).

    cot_ids : group label per report sample (which sampled CoT it came from).
    report_success : 0/1 per report sample.
    ICC ~ 1  -> report near-deterministic given the CoT (ignition: access decided
                in the workspace). ICC << 1 -> the read-out is itself stochastic
                (graded). One-way random-effects ANOVA estimator on binary data.
    """
    cot_ids = np.asarray(cot_ids)
    y = np.asarray(report_success, dtype=float)
    groups = [y[cot_ids == g] for g in np.unique(cot_ids)]
    groups = [g for g in groups if len(g) > 0]
    K = len(groups)
    if K < 2:
        return float("nan")
    N = sum(len(g) for g in groups)
    grand = y.mean()
    ms_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (K - 1)
    ms_within = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(N - K, 1)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (K - 1)
    denom = ms_between + (n0 - 1) * ms_within
    if denom <= 0:
        return 0.0
    return float((ms_between - ms_within) / denom)
