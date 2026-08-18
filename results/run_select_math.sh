#!/bin/bash
# Selection scan (issue #14 MATH ask): which model x MATH-level cells have CoT
# HELPING the load (cot t1 > direct t1)? Lean base-only pilot. n=15/cell.
set -u
cd /Users/ulysse/__projects__
PY=LLM_Blink/.venv/bin/python3
R=LLM_Blink/results
LOADS="math_bench_1 math_bench_2 math_bench_3 math_bench_4"
COMMON="--loads $LOADS --regimes cot direct --n-tasks 5 --n-seeds 15 --base-temp 1.0"
echo "[select] START $(date)"
# Pair 1: two models concurrently (Ollama MAX_LOADED_MODELS=2)
$PY -B LLM_Blink/select_math.py --model qwen2.5:3b $COMMON --out $R/select_qwen2.5_3b.csv > $R/select_qwen2.5_3b.log 2>&1 &
A=$!
$PY -B LLM_Blink/select_math.py --model qwen2.5:7b $COMMON --out $R/select_qwen2.5_7b.csv > $R/select_qwen2.5_7b.log 2>&1 &
B=$!
wait $A; echo "[select] qwen3b done $(date)"
wait $B; echo "[select] qwen7b done $(date)"
# Pair 2
$PY -B LLM_Blink/select_math.py --model llama3.1:8b $COMMON --out $R/select_llama3.1_8b.csv > $R/select_llama3.1_8b.log 2>&1 &
C=$!
$PY -B LLM_Blink/select_math.py --model mistral:7b $COMMON --out $R/select_mistral_7b.csv > $R/select_mistral_7b.log 2>&1 &
D=$!
wait $C; echo "[select] llama done $(date)"
wait $D; echo "[select] mistral done $(date)"
echo "[select] ALL DONE $(date)"
