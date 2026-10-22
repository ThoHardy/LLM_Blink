#!/bin/zsh
# Issue #18 §3.2/§3.3/§3.4 pre-flight: base-draw scan per model to size the
# generation budget (truncation + realised length) and titrate the load rungs
# (direct t1_correct). Uncapped CoT (budget=None), generous max_new_tokens so the
# length tail is unbiased. Runs on tonight's models, ascending size.
set -e
cd /Users/ulysse/__projects__
PY=/Users/ulysse/__projects__/LLM_Blink/.venv/bin/python3
export PATH="/opt/homebrew/bin:$PATH"

LOADS="trivial math_bench_2 math_bench_4 math_bench_5"
SEEDS=15
MNT=8192

for M in qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b llama3.1:8b; do
  echo "===== preflight $M ====="
  $PY -B LLM_Blink/select_math.py --model "$M" \
    --loads ${=LOADS} --regimes cot direct \
    --n-tasks 5 --n-seeds $SEEDS --base-temp 1.0 --max-new-tokens $MNT \
    --out "LLM_Blink/results/preflight_${M//:/_}.csv"
done
echo "===== preflight ALL DONE ====="
