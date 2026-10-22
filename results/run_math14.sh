#!/bin/bash
# Main MATH-load run (issue #14, Thomas's MATH ask): report probe k=20, base T=1,
# access off, n=100/cell. 3 models selected for CoT-HELPS-the-load on MATH loads.
# Sequential @ 8 workers (GPU-bound; seed-major + resumable). Fastest/mixture-
# critical model first so a first figure lands early.
set -u
cd /Users/ulysse/__projects__
PY=LLM_Blink/.venv/bin/python3
R=LLM_Blink/results
COMMON="--n-tasks 5 --loads trivial math_bench_2 math_bench_3 math_bench_4 \
  --regimes cot direct --n-seeds 100 --k 20 --access off --passphrase-last \
  --base-temp 1.0 --n-workers 8"
echo "[math14] START $(date)"
for spec in "qwen2.5:3b:qwen2.5_3b" "qwen2.5:7b:qwen2.5_7b" "llama3.1:8b:llama3.1_8b"; do
  model="${spec%%:*}"; model="${spec%:*}"; tag="${spec##*:}"
  echo "[math14] === $model -> math_$tag START $(date) ==="
  $PY -B LLM_Blink/probe_pilot.py --model "$model" $COMMON \
      --out $R/math_$tag.csv > $R/math_$tag.log 2>&1
  echo "[math14] === $model DONE $(date) ==="
done
echo "[math14] ALL DONE $(date)"
