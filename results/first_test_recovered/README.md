# First AB tests — recovered traces

Recovered 2026-07-17 from Desktop screenshots and notebook outputs. The original
figures/CSVs were written to `/tmp` and never committed, so these images are the
only surviving trace. **Lesson: save runs into the repo, not `/tmp`.**

There were two separate "first" runs:

## 1. `gemma3:4b`, this Mac (M4 Max), Fri 2026-06-05
Two sweeps that evening. The 1000-trial figures were written to `/tmp` and lost;
recovered 2026-07-17 from figures Ulysse had saved to WhatsApp. The 2200-trial
ladder additionally survived as a Desktop screenshot.

- `2026-06-05_gemma3-4b_matrices_1000trials.png` — **1000-trial** sweep, 5 loads
  (none/easy/hard/easy_math/hard_math). Four heatmaps: P(T2 reported) cot|direct,
  T2 mean log-prob cot|direct. (was `/tmp/ab_matrices.png`, from WhatsApp)
- `2026-06-05_gemma3-4b_AB-curves_1000trials.png` — **1000-trial** sweep. The four
  key panels: T2 report vs lag (COT) = "the AB curve", same (DIRECT), Report-blink
  = P(none)−P(load) (COT), and T1 sanity bars. (was `/tmp/ab_full.png`, from WhatsApp)
- `2026-06-05_gemma3-4b_full-ladder_2200trials_heatmaps.png` (Desktop screenshot)
  and `..._2200trials_v2.png` (WhatsApp, same run) — later **2200-trial** FULL
  ladder: 11 conditions (none, semantic_0..4, math_0..4) × 10 lags × {cot,direct}
  × 10 seeds.
- `2026-06-05_session-recap.png` — the Claude Code session; recap reads
  *"the 1,000-trial gemma3:4b sweep is done and plotted (no clean blink; load tasks
  mostly not performed)."*
- **Findings:** direct report at ceiling (1.0) everywhere; cot report falls with
  difficulty; math_1..4 report ≈ 0 (format failure under math load, not a real
  deficit); T1 accuracy below 0.5 for most loads (load not paid). No clean blink.

## 2. Qwen-0.5B, Thomas's Windows laptop (NVIDIA RTX 1000 Ada)
- `first_test_fig1_cell13.png` — extracted from `attentional_blink_local.ipynb`
  outputs. 720-trial none-vs-semantic_4 run. T1 acc on semantic_4 = 0.092 (this is
  the "0.09 on Qwen 0.5B" cited in issue #9). T2 slot-missing 0.275.

Both runs reached the same conclusion the 2026-07-17 smoke test reaches again:
**report saturates at ceiling and the hard T1 load is mostly not performed**, so no
interpretable blink. See `../../MACHINE.md` and the live dashboard.
