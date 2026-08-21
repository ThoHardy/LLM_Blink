#!/bin/zsh
# Issue #18 Campaign A — within-family ladder, fully resampled (figs 2,3,6,7).
# Per model: 8 cells {trivial,math_bench_2,math_bench_4,math_bench_5} x {cot,direct},
# 100 seeds, K=20 FULL regenerations/trial (fork offset 0), T=1, passphrase_last,
# uncapped CoT (budget none), max_new_tokens 8192 (MATH loads; §2.1 safety net).
# Ascending size, commit per model as it lands (done by the operator, not here).
# Resumable: probe_pilot skips completed cells, so re-running continues.
#
# Models: tonight = the low Qwen2.5 ladder. 14b/32b/72b appended for whoever has
# the wall-clock; they will simply continue the same seed-major CSVs.
set -e
cd /Users/ulysse/__projects__
PY=/Users/ulysse/__projects__/LLM_Blink/.venv/bin/python3
export PATH="/opt/homebrew/bin:$PATH"

LOADS="trivial math_bench_2 math_bench_4 math_bench_5"
MODELS=(${=CAMPAIGN_A_MODELS:-qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b})

for M in $MODELS; do
  echo "===== Campaign A: $M  $(date) ====="
  $PY -B LLM_Blink/probe_pilot.py --model "$M" \
    --loads ${=LOADS} --regimes cot direct \
    --n-tasks 5 --n-seeds 100 \
    --resample-full 20 --report-forks 0 --access on \
    --base-temp 1.0 --budget none --max-new-tokens 8192 \
    --n-workers 8 --keep-logs \
    --out "LLM_Blink/results/scale_${M//:/_}.csv"
  echo "===== Campaign A: $M DONE  $(date) ====="
done
echo "===== Campaign A ALL REQUESTED MODELS DONE ====="
