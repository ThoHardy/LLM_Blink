# Night log — issue #14: all-or-none vs graded access

Autonomous session, night of **2026-08-09 → 10**, on Ulysse's M4 Max.
Working the core path of Thomas's ticket #14: **Steps 1, 2, 3** + a local Ollama
pilot. Goal is not just to build tools but to get a first real answer to the
paper's question — is CoT-induced blindness **all-or-none** (ignition, π moves)
or **graded** (resource sharing, μ moves)? — and then iterate.

Branch: `feat/issue-14-mixture-probes`. Coordination comment posted on the issue.

---

## Orientation (what the existing corpus says)

The budget=None CIB corpus (`results/cib_*.csv`, n=100 × 18 models) — the effect
Thomas cites — is **strongly model-dependent** (Direct − CoT `report_contains`,
k=5 tasks, averaged over semantic loads; "cotSemLvl" = cot report level, i.e.
how far off-ceiling the model is):

| model | CIB mean | cot report level | note |
|---|---|---|---|
| gemma2:2b | **+0.36** | 0.50 | strong, off-ceiling → **pilot** |
| qwen2.5:0.5b | +0.31 | 0.02 | strong but near floor |
| mistral:7b | +0.27 | 0.64 | strong, off-ceiling |
| qwen2.5:3b | +0.22 | 0.70 | (memory: "cleanest") |
| 9B+ Gemma/Qwen | ~0.00 | 1.00 | ceiling both regimes — uninformative for a shape claim |
| llama3.2:3b | −0.06 | 0.52 | CoT *helps* (inverted) |

**Pilot choice: gemma2:2b** — the mixture question is testable only where per-trial
report probability spans the range, and gemma2:2b sits near 0.5 with the largest
effect. Ceiling models (the 9B+) carry no shape information.

---

## Step 3 — the analysis (DONE, validated) ✅

`mixture.py`: two-component **beta-binomial mixture** on the per-trial counts
(s, k), parameterised by component mean μ and overdispersion ρ (both in (0,1); ρ
tied across conditions — the modes are stable objects). Four models fit by EM:

- `M0` (1 comp, μ free per cond) — pure graded, no modes
- `Mpi` (μ,ρ tied; π free) — **ignition**
- `Mmu` (π,ρ tied; μ free) — **resource sharing**
- `Mfull` (π,μ free) — unconstrained reference

Model comparison = **held-out predictive log-likelihood** (stratified 10-fold CV,
paired bootstrap over trials — NOT BIC/AIC, NOT a t-test across folds, per §4.1).
Plus Tarone's Z (overdispersion), the §4.2.3 parametric-bootstrap **π→μ null band**
(so μ-drift produced by the estimator itself under a pure-π world is not mistaken
for gradedness), the §5 3-level access taxonomy, and the §4.3 nested ICC.

**Speed:** first implementation used `scipy.stats.betabinom.logpmf` in the EM hot
loop → a single CV did not finish in 10 min. Rewrote the pmf manually with
`gammaln` and precomputed the (s,k)-only binomial term → single Mfull fit 0.15 s,
one CV 2 s. `bb_logpmf` matches scipy to 1e-15.

**Validation (`tests/test_mixture.py`, all 5 pass):** three synthetic generators
of known type, and the analysis reads each correctly — the decider does NOT
prefer either answer:

| generator | verdict | recovered params |
|---|---|---|
| pure π-shift (ignition) | **all-or-none** (Mpi≈Mfull, Mmu<Mfull) | π=[.82,.54,.16] (true [.8,.5,.2]), μ fixed |
| pure μ-shift (resource) | **graded** (Mmu≈Mfull, Mpi<Mfull) | μ slides, π fixed |
| single rate (no modes) | **graded** (M0 beats Mfull on held-out) | Mfull overfits, CV penalises it |

π→μ null band on the pure-π fit: spurious Δμ ∈ [−0.036, +0.034], observed −0.011
→ inside the band. Calibration works.

---

## Next

- Step 1 — `resample.py` forked-resampling primitive (both backends; test on Ollama).
- Step 2 — wire the report/access probes into `experiment.py`.
- Pilot — gemma2:2b, then ANALYSE and iterate (Ulysse's directive: don't just run,
  analyse and launch the next experiment off the result).
