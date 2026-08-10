<!-- DRAFT issue-#14 comment for Ulysse to review and post (not auto-posted:
results are preliminary, n~40/cell). Branch: feat/issue-14-mixture-probes -->

**Update — Steps 1–3 built + validated, and a first two-model result.**

**Built (branch `feat/issue-14-mixture-probes`):**
- **Step 3 · `mixture.py`** — two-component beta-binomial mixture (μ, ρ), M0/Mpi/Mmu/Mfull compared by held-out CV log-lik with paired bootstrap (not BIC/AIC, not cross-fold t-test), plus Tarone's Z, the §4.2.3 π→μ null band, access taxonomy, nested ICC. **Validated** on synthetic generators (`tests/test_mixture.py`): pure-π→ignition, pure-μ→graded, single-rate→M0. It does not prefer either answer.
- **Step 1 · `resample.py`** — `fork_samples` (both backends, distinct per-sample seeds from a separate RNG so `_rebuild` stays bit-exact; degenerate-fork guard). `Trajectory.offset`. Fork point = the `<Final_Answers>` opener (the real read-out transition; `</Thinking>` is emitted on only ~19% of gemma2:2b cot trials).
- **Step 2 · `probe_pilot.py` / `nested_pilot.py`** + analyzers.

**First result (gemma2:2b n≈40/cell, replicated on qwen2.5:3b):**
- **Access probe = graded** (M0 wins the CV; over-dispersion Z≈6; access-rate mass in the middle).
- **Report probe = bimodal** (M0 rejected, Z≈94–112; piles at 0 and k).
- **Nested probe: read-out is decided by the CoT** — ICC 0.73–0.81 (gemma2:2b), 0.85–0.86 (qwen2.5:3b). report | passphrase-in-CoT ≈ 0.9, | not-in-CoT ≈ 0.15–0.34.

→ **graded workspace access, all-or-none read-out gated by workspace entry** — the discreteness arises at the read-out (the §1 second headline). **Mechanism:** CIB is a serial-scan completion failure — CoT length mediates access/report (within-cell ρ up to +0.50).

**Open / honest:** π-vs-μ within the confirmed bimodality is ambiguous at n≈40 (needs the confirmatory n≥200); report has a ~0.15–0.34 context-recovery graded backstop; the load axis is non-monotonic at T=1 on gemma2:2b (short-CoT artifact) — `n_tasks` is the cleaner load knob.

Figures + full trail: `results/fig_two_stage_gemma2_2b.png`, `results/fig_replication_two_stage.png`, `results/NIGHT_LOG_14.md`.
