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

## Step 1 — forked-resampling primitive (DONE) ✅

`resample.py::fork_samples` rebuilds `prompt + traj.text[:fork]` and draws k
continuations at T=1 with DISTINCT per-sample seeds from a SEPARATE numpy
Generator (stimulus RNG untouched → `_rebuild` stays bit-exact). Raises
`DegenerateForkError` if all k are identical (the Ollama fixed-seed silent
failure, §3.4). `model.py`: seed passthrough on both backends. `readout.py`:
`t2_in_cot` (name-or-content access probe). Committed `6676ac5`.

### Hard reality confronted: small models don't emit the clean template

Inspecting real gemma2:2b cot outputs (the pilot model) forced a redesign of the
fork point. gemma2:2b **almost never closes `</Thinking>` (19%) and never emits
`</Final_Answers>`** — it jumps `<Thinking> … <Final_Answers>\n- Task X: …` then
EOS. So the fork point had to be the **`<Final_Answers>` OPENER** (the real
read-out transition, present ~99% on every model tried), not `</Thinking>`.
Forking on `</Thinking>` would have left the primary probe undefined on 80% of
the very trials that carry the effect. The DIRECT regime has NO scaffold at all
(report lines in a bare ``` fence) → its report probe forks at `response`
(offset 0, resample the whole turn) = p(report | prompt).

### The CIB mechanism on gemma2:2b is REAL, not a truncation artifact

Corpus check (unlike the June gemma3:4b "blink"): `output_truncated = 0.0` in
every cell. The effect is genuine — under cot the serial enumeration in
`<Thinking>` sometimes **drops the passphrase task entirely** (it is last / rank
5), consolidating the load tasks and losing the simple one. Exactly the AB
signature (the last, simple item drops when the serial stage is busy). And
`t2_echoed_in_cot` ~0.3–0.46 on cot means the passphrase is often IN the CoT yet
NOT reported → the §5 **level-2 cell (accessed, not reported)** exists in the raw
corpus already.

## Step 2 — probes + pilot runner (DONE) ✅

`probe_pilot.py` (standalone, resumable, incremental, threaded) writes per-trial
`report_s/k` and `access_s/k`. `probe_analyze.py` runs validity + the mixture CV
verdict + histograms. Committed. (Full experiment.py column integration deferred
as a follow-up — the standalone path already delivers the science.)

Live k=10 smoke on gemma2:2b (semantic_2) already suggestive:
- report probe (cot): per-trial rates **0/10, 10/10, 2/10** → looks BIMODAL
  (ignition at the report stage);
- access probe (cot): **6/10, 4/10, 8/10** → intermediate (access looks more
  GRADED). A two-stage dissociation, which is the paper's second headline.

## Pilot (RUNNING)

gemma2:2b, n_tasks=5, loads {trivial, semantic_2, semantic_4}, regimes {cot,
direct}, passphrase_last, base_temp=1.0, k=20, access on, n_seeds=150 → 900 cells,
~4 h, `results/probe_gemma2_2b.csv` (seed-major so partial data spans all cells).
Ollama runs manual `screen -S ollama_par` with OLLAMA_NUM_PARALLEL=8.

## Next (analyse-and-iterate)

- Analyse partial → first π-vs-μ read on report AND access stages.
- Iterate: 2nd model (qwen2.5:3b / mistral:7b), or the D4 report-order axis, or a
  finer load axis — decided by what the first analysis shows.
