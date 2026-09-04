# Issue #18 campaign — progress & findings

*Autonomous overnight run. This file is the human-readable summary of what was
built, what was found, and what is still running. Numbers marked "live" are
regenerated as the ladder fills (`scale_stats.txt`, `fig{2,3,6,7}`).*

## TL;DR for Thomas

1. **§3.1 matcher audit is done and the answer is the good one.** `t2_in_cot` has
   ~0 false negatives; the absent-but-reported trials are a **genuine second route**
   (report without CoT verbalisation), model-dependent 0–5%. `P(report | not in CoT)`
   is therefore clean, not matcher contamination. → `results/matcher_audit.md`.
2. **The read-out you specified is built and tested.** `--resample-full K` draws K
   whole trajectories from offset 0 and scores **both borders on the same K samples**
   (the fig-6 requirement), with a hard per-cell truncation gate (§2.1) and the
   quant tag logged per row (§2.2). Legacy mid-trajectory forks are off.
3. **Pre-flight passed the truncation gate and surfaced a real titration fact:**
   at `--max-new-tokens 8192`, **truncation is 0.0%** on every Qwen cell tested.
   But **sub-7B Qwen is floored on MATH-Algebra** (direct `t1` ≈ 0.02–0.17 on
   `math_bench_2/4/5`), so the small end of the ladder is floor-compressed on the
   math loads — its blink there is an artefact to annotate (like `qwen2.5:0.5b`),
   and the real MATH titration window only opens at 7B+.
4. **Campaign A result (5/6 Qwen rungs in, 32b running):** the blink **shrinks
   with scale and is gone by 14B** — clean-ladder blink at the hardest MATH load
   falls 3b 0.36 → 7b 0.27 → 14b ≈ 0, while within each model it grows with load
   hardness (position-clean, §8.2). This leans **monotone-decreasing**, not an
   interior peak — a legitimate, reportable outcome — with the one nuance that 3b
   could be the peak if the floor-compressed 0.5b/1.5b hide the left-hand rise.
   See the "Result so far" section below.
5. **Campaigns B/C/D:** the tooling for B (position × report-order) is in
   (`--passphrase-rank`, plus the already-wired `--report-order` / `load_engagement`
   T1-ignore arm). C2 (filler reasoning packets) and the 9 D mitigation arms are
   **specified but deliberately not implemented unsupervised** — their exact
   stimulus/prompt wording wants your sign-off (see "Deferred" below).

---

## What was done

### §3.1 — matcher audit (no GPU) — `results/matcher_audit.md`
Read-only pass over 10,800 saved `cot` trials (18 models). The current matcher
codes the passphrase present 86.6% of the time (98.4% of those via the literal
phrase). Of the 13.4% it codes absent, 118 (1.1%) are nonetheless reported. Only
2/118 carry any allusive "copy/three-words" cue, and **both are boilerplate echoes**
(the fixed worked example `VICTOR XRAY ROMEO`, and the system instruction) — not
references to the real passphrase. So there are no meaningful false negatives, and
loosening the matcher would false-*positive* on that boilerplate. **Recommendation:
keep the matcher.** The absent-but-reported route is real (qwen2.5:3b 5.0%,
llama3.1:8b 3.8%, mistral:7b 3.0%) and deserves its own experiment (the dashed
arrow, slide 1) — a proposed design is in the audit file.

### §4 read-out — `--resample-full` (probe_pilot.py, resample.py)
Per trial: K complete regenerations from the empty assistant prefix (fork offset
0), T=1; `s_access/K` = passphrase in `<Thinking>`, `s_report/K` = passphrase in
`<Final_Answers>`, both off the **same** K trajectories. Per-sample vectors
exported (`report_bits`, `access_bits`, `t1_per`, `cot_len_per`) so figs 2 and 6
are possible. Direct regime logs `s_access = None` (no CoT). §2.1 gate: per-sample
truncation tracked, per-cell rate printed with a >2% FAILED flag. §2.2: Ollama
quant tag per row. §7.2: `--keep-logs` writes every transcript to
`<out>.samples.jsonl` (gzipped on commit); `raw_output` kept for a 200-trial
subsample in the CSV. Tests: `tests/test_resample_full.py`.

### §3.2/§3.3/§3.4 — pre-flight — `results/preflight_report.txt`
Base-draw scan over the four Campaign-A rungs at 8192. Findings (Qwen 0.5–3B):

| model | trunc (all cells) | cot len p99 | MATH direct t1 | verdict |
|---|---|---|---|---|
| qwen2.5:0.5b | 0.0% | 3702 ch (~1028 tok) | ~0.00 | floor on all math |
| qwen2.5:1.5b | 0.0% | 5381 ch (~1495 tok) | 0.02–0.03 | floor on all math |
| qwen2.5:3b | 0.0% | 2360 ch (~656 tok) | 0.03–0.17 | floor on all math |

Truncation is a non-issue at 8192. The titration verdict is the substantive one:
**MATH-Algebra L2–L5 is too hard for sub-7B Qwen**, so those cells are
floor-compressed. They still anchor the low end of fig 3 (annotated) and the
left end of the fig-6 bimodality prediction (unimodal-low entry), and the
`trivial` rung + passphrase-position contrast remain informative at every size.

### §5 tooling — `--passphrase-rank` (Campaign B)
`TrialConfig.passphrase_rank` fixes the passphrase's 1-indexed stream rank,
generalising `passphrase_last`; byte-exact when unset. With the already-wired
`--report-order {stream,reverse}` and `load_engagement=ignore` (the T1-ignore
control), Campaign B is runnable. Test: `tests/test_passphrase_rank.py`.

---

## Campaign A — status (live)

Qwen2.5 ladder, ascending, 8 cells {trivial, math_bench_2/4/5} × {direct, cot},
100 seeds, K=20, T=1, `passphrase_last`, uncapped CoT, `--max-new-tokens 8192`,
`--keep-logs`. Committed per model as it lands. See `scale_stats.txt` (truncation
gate first) and `fig3_blink_vs_size.png` / `fig2` / `fig6` / `fig7`.

**Pace reality (important).** On this Mac each trial costs base + 20 resamples,
batched 8-wide (server `OLLAMA_NUM_PARALLEL=8`): ≈ 13 s/trial for `0.5b`, ≈ 30
s/trial for `1.5b` (it rambles longer on MATH). So a single model is 3–7 h of
wall-clock and **the full Qwen ladder is a multi-night job** — exactly the §9
expectation (~96 k generations for Qwen alone). The orchestrator runs ascending
and commits each model as it finishes, so the ladder is readable as it fills and
resumes across nights. **The scientifically informative MATH points are 7B+**
(sub-7B is math-floored, §3.4), so the first night yields the floor anchors and
the pipeline proof; the rising/peak region lands on subsequent runs. *(Update: on
day 3, with Ulysse's OK, the server was restarted at `OLLAMA_NUM_PARALLEL=16` with
`OLLAMA_CONTEXT_LENGTH=12288` — the prompt is only ~971 tok so a 32k context was
pure waste; shrinking it freed enough KV cache to double parallelism while cutting
14b memory 60→47 GB. ~1.3× faster and much safer memory for the 32b rung.)*

### Result so far — the blink shrinks with scale and is GONE by 14B

Five of six Qwen2.5 rungs are in (0.5b → 14b); **32b is running** (~half done).
All passed the §2.1 truncation gate (worst cell ≤ 1.16 %, most 0.00 %). Blink =
`report_direct − report_cot`, `passphrase_last` held fixed:

| model | blink trivial | blink math_2 | blink math_4 | **blink math_5** | clean? |
|---|---|---|---|---|---|
| 0.5b | 0.32 | 0.25 | 0.21 | 0.21 | **floor artefact** |
| 1.5b | −0.02 | 0.11 | 0.14 | (floor) | **floor artefact** |
| 3b | 0.10 | 0.25 | 0.26 | **0.36** | clean |
| 7b | 0.02 | 0.22 | 0.24 | **0.27** | clean |
| 14b | 0.007 | ~0 | ~0 | **−0.002** | clean |

**Two things to read off this:**

1. **Within-model, blink grows with load hardness** (e.g. 3b: 0.10 → 0.36 as MATH
   gets harder, passphrase position identical) — the §8.2 position-clean evidence
   for capacity competition. Holds at every clean size.
2. **Across scale, the blink falls monotonically over the clean points**
   (3b 0.36 → 7b 0.27 → 14b ≈ 0) and is **abolished by 14B**: CoT no longer costs
   the passphrase at all (14b math_5 cot 0.994 vs direct 0.992).

**Floor caveat (§3.4).** 0.5b and 1.5b report the passphrase < 0.8 even in `direct`
(compressed ceiling), so their apparent blink is an artefact, annotated, *not* a
real rise on the left of the curve.

**What this says about the §4 headline prediction.** The prediction was an
*interior peak* (capacity competition). The clean data are **monotone decreasing
from the smallest clean model (3b) to zero at 14b** — which, per the issue's own
falsification table, leans toward *"large models just follow instructions
better"* rather than an interior peak, **unless** 3b is itself the intermediate
peak with the true left-hand rise hidden under the floor-compressed 0.5b/1.5b
(unmeasurable cleanly). Either reading is a legitimate, reportable outcome (the
issue's "hard caveat" allows for it). **32b will confirm the tail** (expected ≈ 0);
whether the peak is "interior at 3b" or "monotone" is the one open nuance, and it
turns on the floor models we cannot de-confound on the report axis.

*(`scale_stats.txt` is the authoritative live table; `fig3_blink_vs_size.png` the
curve. Both auto-refresh per model.)*

---

## Deferred (specified, awaiting your sign-off before running)

**Why deferred:** these need stimulus/prompt wording choices that change what the
experiment measures, and the issue itself asks for a *pre-registered predicted
ordering* (D) — better decided with you than fixed unilaterally overnight.

- **C2 — filler reasoning packets (§6).** Proposed mechanism: add a `filler_reasoning`
  packet kind (reasoning content, **no `Task` tag**) inserted between the last load
  task and the passphrase, count ∈ {0,1,2,4}, exogenous ⇒ instrument for realised
  CoT length. Open question worth one pilot: an *untagged* reasoning packet may be
  skipped by the model (the instructions say skip fillers), so it might not lengthen
  the CoT — we should verify the manipulation "bites" on 20 trials before scaling.
  C1 (regress report on the realised `cot_len_per` already logged by Campaign A) is
  **free and needs no new code** — it can be run on Campaign A's output immediately.
- **D — 9 mitigation arms (§7).** D4 (report simple item first) = `--report-order
  reverse` (already in); D5/T1-ignore ≈ `load_engagement`. D1/D2/D3/D6/D8/D9 each need
  a specific prompt-string variant; D7 is a second call. Each is one `--mitigation
  <arm>` branch in `stimuli.py`. Recommend fixing the exact wording with you, then
  running all arms paired on the Campaign-A `cot` seeds at the peak load.

---

## Pitfalls (issue §10) — how each is handled

1. Silent truncation → per-sample flag + per-cell gate printed first; 0% at 8192.
2. Matcher false negatives → audited, ~0; matcher kept.
3. Floor models read as blinking → titration flags them; sub-7B Qwen annotated floor.
4. Mixed quant tags → quant tag logged per row (`quant_tag`), all Q4_K_M so far.
5. Resampling at the wrong place → `--resample-full` forks at offset 0 (whole turn),
   not mid-trajectory; legacy report-forks off.
6. Length confounded with load → C1 free now; C2 designed.
7. Publishing the leaderboard before Campaign B → fig 7-left withheld until B runs.
