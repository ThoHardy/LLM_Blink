#!/bin/zsh
# Issue #18 Campaign A extension — Gemma3 + Mistral ladders.
# Run ONE model, then gzip its transcript and regenerate the (multi-family)
# figures across every scale_*.csv on disk. Commit + push + artifact are done by
# the operator between models, so this script is side-effect-free w.r.t. git.
#
# Usage:  run_family_model.sh <ollama-model> <out-slug>
#   e.g.  run_family_model.sh gemma3:270m gemma3_0.27b
#
# Resumable: probe_pilot skips completed cells, so re-running continues.
set -e
MODEL="$1"; SLUG="$2"
if [[ -z "$MODEL" || -z "$SLUG" ]]; then echo "usage: $0 <model> <slug>"; exit 2; fi
cd /Users/ulysse/__projects__
PY=/Users/ulysse/__projects__/LLM_Blink/.venv/bin/python3
export PATH="/opt/homebrew/bin:$PATH"
R=LLM_Blink/results
LOADS="trivial math_bench_2 math_bench_4 math_bench_5"

echo "===== Campaign A ext: $MODEL -> $SLUG  START $(date) ====="
$PY -B LLM_Blink/probe_pilot.py --model "$MODEL" \
  --loads ${=LOADS} --regimes cot direct \
  --n-tasks 5 --n-seeds 100 \
  --resample-full 20 --report-forks 0 --access on \
  --base-temp 1.0 --budget none --max-new-tokens 8192 \
  --n-workers 16 --keep-logs \
  --out "$R/scale_${SLUG}.csv" >> "$R/campaignA_${SLUG}.log" 2>&1

# compress the transcript (text compresses ~6-10x) so it is committable
[ -f "$R/scale_${SLUG}.samples.jsonl" ] && gzip -f "$R/scale_${SLUG}.samples.jsonl"

# refresh the combined multi-family figures + stats over everything on disk
$PY -B LLM_Blink/make_scale_figures.py $R/scale_*.csv >> "$R/campaignA_${SLUG}.log" 2>&1

echo "===== Campaign A ext: $MODEL -> $SLUG  DONE $(date) ====="
