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

## FIRST REAL RESULT — gemma2:2b, n=30/cell (partial pilot)

Validity all green: report fork coverage 1.00, degenerate rate 0.00 (seeds
effective), calibration mean(report_rate)≈realized binary, **self-consistency
0.91 (realized=1) vs 0.16 (realized=0)** — the probe measures a real,
CoT-content-dependent quantity (0.16 = context-recovery floor).

**1. ACCESS = GRADED (clean).** Condition=load, cot. M0 (single graded component)
beats every mixture: M0−Mfull +0.071, CI [0.028, 0.120] excludes 0; Tarone Z=6.1
(mild). Access rates cluster mid-range (~0.4), no piling at 0/k. → **entry into
the serial workspace is a graded / resource-limited process, not all-or-none.**

**2. REPORT = strongly overdispersed (Tarone Z=94.5), modes near 0.05/0.95**
(ignition-shaped), but **underpowered at n=30**: Mfull does not yet beat M0
(a single high-ρ Beta can mimic a U-shape — exactly the subtlety §4 warns of).
π→μ null test marginal. Needs n≈120 to separate bimodal vs graded.

**3. TWO-STAGE DISSOCIATION visible** (the paper's 2nd headline): overdispersion
6 (access) vs 94 (report) → any discreteness is concentrated at the **read-out**
stage, not at workspace entry. Emerging answer to "where does discreteness
arise": at report, not access.

**4. Anomaly — inverted load gradient at T=1.** trivial-cot report 0.37 <
semantic-cot 0.53–0.59 — INVERTED vs the T=0 corpus (trivial 0.74 > semantic
0.5). D6 within-cell: longer CoT → MORE report (Spearman ρ=+0.40, p=0.03 on
semantic_4). → the CoT-blindness is mediated by CoT length/thoroughness; trivial
load → short CoT → drops the (last) passphrase more. The CIB effect (direct−cot)
is LARGEST for trivial at T=1 (0.91→0.37). To investigate.

**5. Access taxonomy:** level-2 (accessed, NOT reported) ~0.15–0.20 across loads —
the theoretically-loaded GWT cell is populated.

## DECISIVE RESULT — the two-stage dissociation (gemma2:2b)

`results/fig_two_stage_gemma2_2b.png`. Three independent measures agree:

**A. Report probe is BIMODAL** (n=126 cot pooled): counts pile at 0 AND k. The
mixture on report counts (condition=regime, n=252) rejects M0 decisively —
M0−Mfull −0.178, CI [−0.266, −0.085] excludes 0; Tarone Z=112. So report is NOT
a single graded component: **two modes (~0.08 / 0.85)**.

**B. Access probe is UNIMODAL, GRADED** (mass in the middle, peak ~8/20). Mixture:
M0 (single graded component) WINS — M0−Mfull +0.071, CI [0.028, 0.120]; Tarone
Z=6.1. Workspace ENTRY is graded/resource-limited, not all-or-none.

**C. Nested probe: report is decided by the CoT** — ICC 0.73 (semantic_4), 0.81
(trivial); between-CoT var (0.19–0.21) ≫ within-CoT var (0.04–0.06). report | in
CoT = 0.90–0.94, report | not in CoT = 0.30–0.34 (a context-recovery channel).

→ **The discreteness of conscious access arises at the workspace→report READ-OUT,
gated by whether the item entered the serial workspace; workspace ENTRY itself is
graded.** This is a clean, novel two-stage answer to the paper's §1 second
headline, and the LLM analog of the access-vs-report question.

**π-vs-μ (finer):** within the confirmed bimodality, whether load/regime moves π
(fraction in the reported mode) or μ (mode locations) is AMBIGUOUS at n≈126
(Mpi≈Mmu≈Mfull). Under the fixed-mode model the cot→direct CIB reads as a
π-shift (0.54→0.97 in the high mode) — ignition-consistent — but μ-shift is not
excluded. Needs the confirmatory n≥200. NB the regime contrast has a fork-point
confound (cot=post_cot vs direct=response); do not over-read it.

**Caveats / open (honest):** (i) the load axis is non-monotonic at T=1 — trivial
cot report (0.37) < semantic (0.53–0.59), INVERTED vs the T=0 corpus; D6 shows
longer CoT → more report, so the effect is CoT-length/thoroughness mediated. The
cleaner π-vs-μ manipulation is probably n_tasks, not load-difficulty. (ii) report
isn't PURELY all-or-none — the ~0.33 context-recovery channel is a genuine graded
backstop (the passphrase persists in context, §5 scope limit a).

## Plan from here (analyse-and-iterate)

1. Replicate the two-stage dissociation on **qwen2.5:3b** (nested probe → access
   rate + ICC in one run) [RUNNING].
2. If time: mistral:7b; a clean π-vs-μ run with condition=n_tasks; investigate
   the T=1 trivial inversion.
