#!/bin/bash
# Sequential smoke-test driver for the Gemma fleet.
# Each model: n_seeds=10, defaults (lags 0-10, loads none+semantic_4, cot+direct).
# Writes one CSV + one log per model, plus a timing line to fleet_timing.txt.
set -u
ROOT=/Users/ulysse/__projects__/LLM_Blink
PY=$ROOT/.venv/bin/python
cd "$ROOT/.." || exit 1

TIMING=$ROOT/results/fleet_timing.txt

for spec in "$@"; do
  model="$spec"
  slug=$(echo "$model" | sed 's#[/:]#_#g')
  csv="LLM_Blink/results/ab_${slug}_t0_s10.csv"
  log="LLM_Blink/results/smoke_${slug}.log"
  echo ">>> $(date '+%H:%M:%S') starting $model"
  start=$(date +%s)
  "$PY" LLM_Blink/run_experiment.py --model "$model" --n-seeds 10 --output "$csv" > "$log" 2>&1
  code=$?
  end=$(date +%s)
  echo "$model  exit=$code  elapsed=$((end-start))s" >> "$TIMING"
  echo ">>> $(date '+%H:%M:%S') done $model exit=$code elapsed=$((end-start))s"
done
echo ">>> ALL DONE"
