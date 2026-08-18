#!/bin/bash
set -u
cd /Users/ulysse/__projects__
PY=LLM_Blink/.venv/bin/python3
R=LLM_Blink/results
LOADS="math_bench_1 math_bench_2 math_bench_3 math_bench_4"
COMMON="--loads $LOADS --regimes cot direct --n-tasks 5 --n-seeds 8 --base-temp 1.0"
echo "[select2] START $(date)"
$PY -B LLM_Blink/select_math.py --model mistral:7b $COMMON --out $R/select_mistral_7b.csv > $R/select_mistral_7b.log 2>&1 &
A=$!
$PY -B LLM_Blink/select_math.py --model llama3.1:8b $COMMON --out $R/select_llama3.1_8b.csv > $R/select_llama3.1_8b.log 2>&1 &
B=$!
wait $A; echo "[select2] mistral done $(date)"
wait $B; echo "[select2] llama done $(date)"
echo "[select2] ALL DONE $(date)"
