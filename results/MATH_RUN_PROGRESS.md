# MATH-load CIB run (issue #14) — live progress

Follow-up to the confirmatory n=200 run (PR #16). That run reproduced the
CoT-induced blindness at T=1 but on **semantic** loads the *"CoT-helps-the-load"*
leg was flat (CoT did not raise T1 accuracy), so the selective paradox rested on
the blindness leg alone. This run fixes that by moving the load onto the
**Hendrycks MATH-Algebra** benchmark (`math_bench_*`, Levels 1–5), where CoT
genuinely raises load accuracy — making the dissociation airtight:
**CoT helps solve the load, yet blinds the co-present passphrase.**

## Model × load selection (`select_math.py`)

A lean base-only pilot (no k-fork) measured T1 accuracy Direct vs CoT on MATH
levels 1–4 for four candidates. Kept the models where **CoT clearly helps T1**:

| model | MATH T1 Direct→CoT (sampled) | decision |
|---|---|---|
| qwen2.5:3b | L3 0.08→0.42, L2 0.25→0.58 | **keep** (all-or-none read-out) |
| qwen2.5:7b | L3 0.12→0.75, L2 0.42→0.83 | **keep** (strong lift) |
| llama3.1:8b | L1 0.12→0.38, L2 0.38→0.50 | **keep** (other family) |
| mistral:7b | L2 0.38→0.38 (flat) + slow | **drop** (no CoT-help on MATH) |

## Main run (`run_math14.sh`, screen `math14`)

- 3 models × {`trivial`, `math_bench_2`, `math_bench_3`, `math_bench_4`} × {cot, direct}
- **n=100/cell**, report probe **k=20**, base **T=1**, access probe off, 8 workers
- Sequential (qwen2.5:3b first); `probe_pilot.py` is resumable + seed-major
- Raw output: `results/math_{qwen2.5_3b,qwen2.5_7b,llama3.1_8b}.csv`

## Figures (`make_math_figures.py`)

1. `fig_math_reproduce.png` — CoT HELPS T1 yet HURTS passphrase report (per model)
2. `fig_math_per_model.png` — report_rate vs load + cot report-count histogram
   (all-or-none vs graded shape)
3. `fig_math_mixture_mu_pi.png` — Mfull beta-binomial fit: component means
   μ_low/μ_high and high-mode weight π vs MATH load (π-shift = ignition /
   all-or-none; μ-slide = graded / resource-sharing)
4. `math_stats.txt`

## Status

| model | rows / 800 | state |
|---|---|---|
| qwen2.5:3b | 800 | ✅ done |
| qwen2.5:7b | in progress | running |
| llama3.1:8b | 0 | queued |

CSVs + figures are committed **per model as each finishes**, so this PR fills in
progressively; figures are regenerated from all completed models at each step.

### qwen2.5:3b (n=100/cell) — done

The paradox holds cleanly on the standard MATH benchmark:

| load | T1 acc Direct→CoT | passphrase report Direct→CoT |
|---|---|---|
| math_bench_2 | 0.15 → **0.33** (CoT helps) | 0.91 → **0.69** (blink) |
| math_bench_3 | 0.14 → **0.30** | 0.89 → **0.57** |
| math_bench_4 | 0.07 → **0.19** | 0.88 → **0.61** |

Read-out shape **U-shaped / all-or-none** (73% of cot trials at the extremes),
ignition gap **+0.52** (report|in-CoT 0.84 vs |not-in-CoT 0.32), Tarone Z=182.
Mixture (Mfull): the report mode **μ_high stays put** across load (0.88/0.89/0.84)
while **π shifts** (0.75/0.52/0.71) — the ignition (all-or-none) signature.
