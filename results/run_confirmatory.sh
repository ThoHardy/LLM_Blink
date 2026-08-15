#!/bin/bash
# Confirmatory run for issue #14 (Thomas's ask #3): n=200/cell, 3 models.
# Report probe k=20 at base T=1 (plan 3.5); access probe OFF in this pass
# (binary t2_in_cot still logs the access/report dissociation; the graded
# access-stage measure is a smaller follow-up run). Resumable + incremental.
#
# gemma2:2b + qwen2.5:3b run concurrently (Ollama holds 2 models, 8 parallel
# slots -> 4 workers each); mistral:7b runs alone after, at 8 workers.
set -u
cd /Users/ulysse/__projects__
PY=LLM_Blink/.venv/bin/python3
R=LLM_Blink/results
COMMON="--n-tasks 5 --loads trivial semantic_2 semantic_4 --regimes cot direct \
  --n-seeds 200 --k 20 --access off --passphrase-last --base-temp 1.0"

echo "[confirm] START $(date)"

# --- Group 1: the two small (mixture-critical) models, concurrent, 4 workers ---
$PY -B LLM_Blink/probe_pilot.py --model gemma2:2b   $COMMON --n-workers 4 \
    --out $R/confirm_gemma2_2b.csv   > $R/confirm_gemma2_2b.log   2>&1 &
P1=$!
$PY -B LLM_Blink/probe_pilot.py --model qwen2.5:3b  $COMMON --n-workers 4 \
    --out $R/confirm_qwen2.5_3b.csv  > $R/confirm_qwen2.5_3b.log  2>&1 &
P2=$!
echo "[confirm] group1 pids: gemma2=$P1 qwen=$P2"
wait $P1; echo "[confirm] gemma2 done $(date)"
wait $P2; echo "[confirm] qwen done $(date)"

# --- Group 2: mistral alone, 8 workers ---
$PY -B LLM_Blink/probe_pilot.py --model mistral:7b  $COMMON --n-workers 8 \
    --out $R/confirm_mistral_7b.csv  > $R/confirm_mistral_7b.log  2>&1
echo "[confirm] mistral done $(date)"
echo "[confirm] ALL DONE $(date)"
