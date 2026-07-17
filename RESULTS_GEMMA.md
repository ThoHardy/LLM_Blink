# Gemma sweep — results (issue #9)

> Status: **template — no results yet.** Fill each section as runs complete.
> Meta-parameters: temperature 0, loads (none, semantic_4), lags (0 2 4 6 8 10),
> regimes (cot, direct), max_new_tokens 1024, backend Ollama.

## 1. Models

| Model (ollama tag) | Params | Family | Notes |
|---|---|---|---|
| _e.g. gemma3:270m_ | 0.27 B | gemma-3 | |

## 2. Smoke test (n_seeds = 10)

Posted in issue #9 before the full run.

| Model | s / trial | est. full-run time (s1000) | output_truncated | t2_slot_missing | placeholder rate (cot) |
|---|---|---|---|---|---|

**T1 accuracy per load** (was the load actually paid? — flag models near 0 on semantic_4):

| Model | t1_correct (none) | t1_correct (semantic_4) |
|---|---|---|

## 3. Quality gates & retention (full run)

Gates: `output_truncated == False`, `t2_slot_missing == False`, cot rows also
`thinking_is_placeholder == False`. Per-cell retention from
`python leaderboard.py results/ab_*_s1000.csv --retention`:

_(paste tables here)_

## 4. AB curves per model

Both raw and gated (`df_ok`) versions, as at the end of
`attentional_blink_local.ipynb`: `report_correct` vs lag and mean T2 log-prob
vs lag, one line per load, one panel per regime.

_(one subsection per model, smallest to largest)_

## 5. Leaderboard

`python leaderboard.py results/ab_*_s1000.csv --out-table results/leaderboard.csv --out-plot results/blinkscore_vs_params.png`

_(paste table — model, params, regime, LoadCost, BlinkScore, *_lp secondary
columns — ranked by BlinkScore in cot, plus the BlinkScore-vs-params figure)_

## 6. Example trials

Raw examples opened around whatever pattern shows up (`show_trial` in the
notebook): what did the models actually do?

## 7. Interpretation notes

Dip / flat gap / sparing / inverted / null — and hypotheses for follow-ups.
Remember: a null is a legitimate outcome; known confounds (absolute T2
position, decorative CoT) are deliberately left for a later round.
