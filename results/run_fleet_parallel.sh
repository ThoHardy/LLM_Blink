#!/bin/bash
# Parallel smoke-test driver: one model at a time, 8-way intra-model batching.
set -u
ROOT=/Users/ulysse/__projects__/LLM_Blink
PY=$ROOT/.venv/bin/python
WORKERS=${WORKERS:-8}
SEEDS=${SEEDS:-10}
cd "$ROOT/.." || exit 1
TIMING=$ROOT/results/fleet_timing_parallel.txt

for model in "$@"; do
  slug=$(echo "$model" | sed 's#[/:]#_#g')
  csv="LLM_Blink/results/ab_${slug}_t0_s${SEEDS}.csv"
  log="LLM_Blink/results/smoke_${slug}_par.log"
  echo ">>> $(date '+%H:%M:%S') starting $model (workers=$WORKERS seeds=$SEEDS)"
  start=$(date +%s)
  "$PY" LLM_Blink/results/parallel_sweep.py --model "$model" --workers "$WORKERS" \
        --n-seeds "$SEEDS" --output "$csv" > "$log" 2>&1
  code=$?
  end=$(date +%s)
  echo "$model  exit=$code  elapsed=$((end-start))s  ($(echo "scale=2;($end-$start)/(240*$SEEDS/10)"|bc)s/trial)" >> "$TIMING"
  echo ">>> $(date '+%H:%M:%S') done $model exit=$code elapsed=$((end-start))s"
done
echo ">>> FLEET PARALLEL DONE"
