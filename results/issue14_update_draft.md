<!-- DRAFT issue-#14 comment for Ulysse to review and post (not auto-posted:
results are preliminary, n~40/cell). Branch: feat/issue-14-mixture-probes -->

**Update — Steps 1–3 built + validated, and a first two-model result.**

**Built (branch `feat/issue-14-mixture-probes`):**
- **Step 3 · `mixture.py`** — two-component beta-binomial mixture (μ, ρ), M0/Mpi/Mmu/Mfull compared by held-out CV log-lik with paired bootstrap (not BIC/AIC, not cross-fold t-test), plus Tarone's Z, the §4.2.3 π→μ null band, access taxonomy, nested ICC. **Validated** on synthetic generators (`tests/test_mixture.py`): pure-π→ignition, pure-μ→graded, single-rate→M0. It does not prefer either answer.
- **Step 1 · `resample.py`** — `fork_samples` (both backends, distinct per-sample seeds from a separate RNG so `_rebuild` stays bit-exact; degenerate-fork guard). `Trajectory.offset`. Fork point = the `<Final_Answers>` opener (the real read-out transition; `</Thinking>` is emitted on only ~19% of gemma2:2b cot trials).
- **Step 2 · `probe_pilot.py` / `nested_pilot.py`** + analyzers.

**Result — 3 models (gemma2:2b, qwen2.5:3b n≈40/cell; mistral:7b n≈15/cell):**
- **Access is graded on all three** (M0 wins the CV; over-dispersion Z≈6; access-rate mass in the middle).
- **The access/report dissociation is real** (a populated "accessed, not reported" level-2 cell on all three).
- **The read-out's discreteness is MODEL-DEPENDENT:**
  - gemma2:2b, qwen2.5:3b → **all-or-none read-out gated by workspace entry** (nested ICC 0.73–0.86; report | in-CoT ≈0.9, | not-in-CoT ≈0.15–0.34; report distribution bimodal, M0 rejected, Z≈94).
  - mistral:7b → **graded read-out** (ICC 0.12–0.27; report | in-CoT only 0.45–0.66 despite access 0.75–0.96). Mechanism is legible in the raw CoT: mistral writes the passphrase out in `<Thinking>` then *explicitly decides* not to report it ("this task is not identified as such… I will not include it").

→ *Both* of the outcomes you flagged as publishable co-occur across models: workspace access is graded everywhere; the discreteness sits at a different **locus** per model (entry-gating vs a stochastic read-out decision). A 3rd model was essential — two alone would have over-claimed a universal all-or-none read-out. **Mechanism (gemma/qwen):** CIB is a serial-scan completion failure — CoT length mediates access/report (within-cell ρ up to +0.50).

**Open / honest:** π-vs-μ within the bimodal models is ambiguous at n≈40 (needs the confirmatory n≥200); report has a ~0.15–0.34 context-recovery graded backstop; the load axis is non-monotonic at T=1 on gemma2:2b (short-CoT artifact) — `n_tasks` is the cleaner load knob; mistral n is only ~15/load so far (clear qualitatively, worth more n).

Figures + full trail: `results/fig_two_stage_gemma2_2b.png`, `results/fig_replication_two_stage.png`, `results/NIGHT_LOG_14.md`.
