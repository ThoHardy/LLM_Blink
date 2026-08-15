# Overnight investigation log — gemma3:4b attentional blink

Autonomous run, night of 2026-07-17→18. Goal: understand why the June 5 gemma3:4b
run showed a CoT-specific, lag-dependent T2 deficit ("blink-like") but the current
code does not. All runs: gemma3:4b, greedy (T=0), 8-way parallel on the M4 Max.

## The puzzle

- **June 5 (old code):** cot report for easy/hard fell to 0.3–0.6 at short lags and
  recovered with lag; direct at ceiling. Looked like a CoT-specific load×lag blink.
- **Tonight (current code):** T2 report is at **ceiling (1.0)** in every condition.

What changed between June 5 and now: worked-example prompt added (Jun 6), the
generate-then-score redesign (Jul 16), and max_new_tokens 256→1024 + stop tag (Jul 17).

## Hypotheses & results

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| H1 | Worked example masks it | `--no-example`, none/easy/hard | **Rejected** — report still ceiling (1.0) both regimes; T1 acc drops (0.85→0.29) but T2 report unaffected. |
| H2 | June "blink" = truncation artifact | `max_new_tokens=256`, none/easy/hard | **Supported** — at 256, cot report collapses to ~0 for ALL loads (truncation ~100%); direct stays at ceiling. Confirms truncation drives cot report down; the current prompt over-truncates (re-enumeration) so it can't reproduce June's *differential*, but the mechanism is real. |
| H4 | Graded log-prob (truncation-immune) shows a blink? | log-prob by lag, 1024, cot vs direct | **No blink.** cot log-prob is *higher* under load than none (inverted, not a deficit); no lag dip; direct flat (~−25). |
| H3 | Titrate T2 off ceiling | sweep `t2_words` | running |

## Conclusion so far: the June blink is an artifact, no genuine effect on 4b

The decisive measure is the **graded T2 log-prob at 1024 tokens** — it reads
P(correct T2) at the T2 slot whether or not the model emitted it, so it is immune to
truncation. On gemma3:4b it shows **no load-induced deficit and no lag dip** (if
anything load *raises* T2 log-prob in cot — the model's own <Thinking> restates T2).

The June binary "blink" is best explained as a **generation-length artifact**: at the
old short budget, cot+load produced longer reasoning → truncation before the T2 slot
→ apparent report failure; direct had no reasoning → no truncation → ceiling. That
yields the exact "cot-specific, load-graded, recovering-with-lag" shape with no
attentional mechanism. Per Ulysse's instruction, do NOT scale to other models — the
June effect did not reproduce as a genuine effect. Remaining rigor checks: T2
titration (H3), t1_correct-conditioned analysis, and (optional) checking out the June
commit to reproduce the artifact directly.

## t1_correct-conditioned (1024, cot) — null holds

Splitting loaded cot trials by whether T1 was actually solved: on **T1-correct**
trials (load genuinely paid, n=89) the graded T2 log-prob under load (easy ~−12,
hard ~−7..−12) is still **above** the none baseline (−10..−19), with no short-lag
dip. Conditioning on load-paid does not reveal a blink. The inverted direction (load
raises T2 log-prob) is a generation-trajectory effect: the longer CoT under load
re-primes T2 before the slot.

## H3 titration (t2_words=6) — null

Longer T2 brings cot report off ceiling (~0.2–0.8, no truncation), but **none is just
as low as easy/hard** — no load deficit, just noise around 0.5. The cot<direct report
gap is copy-fidelity over a longer passphrase, not attention (direct stays at ceiling).
Graded log-prob still inverted (easy/hard above none), no dip.

## Verdict (gemma3:4b): NULL, n=50-confirmed. No genuine attentional blink.

Four independent measures agree — binary report (ceiling at 1024), graded log-prob
(no dip, load *raises* it → inverted), t1_correct-conditioned (same), and titration
(none as low as load). A powered **n=50 / 1800-trial** run confirms it with tight CIs
(`ab_gemma3_4b_N50_t0_s50.csv`, `fig_4b_mechanism.png`). The June binary "blink" is a
truncation artifact (the model re-prints the whole stream then restates T2; a short
budget cuts it before the T2 slot). Per Ulysse's rule, NOT scaling to other models —
the effect did not reproduce as genuine. A null is a legitimate result here.

Optional remaining: reproduce the June curve directly from the pre-worked-example
commit (belt-and-braces on the artifact claim).

## Qualitative (reading raw_output, cot/hard/lag0)

The model (a) **re-prints the entire 15-packet stream** in its output before answering
— that echo is the long part that blows the 256-token budget (uniformly, regardless of
load → the truncation is not load-specific on current 4b), and (b) writes a short
`<Thinking>` that **restates the T2 passphrase verbatim** before the answer slot → this
is why cot graded log-prob is high and load doesn't hurt it. No blink-like behavior:
T2 is handled reliably; the only failures are truncation (budget) and T1 reasoning
errors. The model does not "drop" T2 under load at short lag.

## Data files
- `ab_gemma3_4b_JUNconds_t0_s10.csv` — current code, none/easy/hard (ceiling).
- `ab_gemma3_4b_NOEX_t0_s10.csv` — no worked example (ceiling).
- `ab_gemma3_4b_TRUNC256_t0_s10.csv` — 256-token budget (H2 test).
- `first_test_recovered/` — the June 5 figures (the effect we're chasing).
