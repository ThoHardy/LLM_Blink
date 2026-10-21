#!/bin/zsh
# Order / target-kind conditions (2026-10-21) for ONE model, Campaign-A protocol.
# Writes one CSV per condition: results/cond_<condition>_<slug>.csv (NOT scale_*,
# so make_scale_figures.py keeps reading the baseline campaign only).
#
# Usage:  run_conditions_model.sh <ollama-model> <out-slug> [condition ...]
#   e.g.  run_conditions_model.sh qwen2.5:3b qwen2.5_3b
#         run_conditions_model.sh qwen2.5:3b qwen2.5_3b rank1 digits
# Conditions: rank1 rank1_loadfirst digits trivmath   (N_MAIN seeds)
#             rank1_exPF rank1_loadfirst_exPF          (N_CTRL seeds, example control)
#
# Resumable: probe_pilot skips completed cells, so re-running continues.
set -e
MODEL="$1"; SLUG="$2"; shift 2 || true
if [[ -z "$MODEL" || -z "$SLUG" ]]; then echo "usage: $0 <model> <slug> [condition ...]"; exit 2; fi
CONDS=("$@")
(( ${#CONDS} )) || CONDS=(rank1 rank1_loadfirst digits trivmath rank1_exPF rank1_loadfirst_exPF)
N_MAIN=${N_MAIN:-100}
N_CTRL=${N_CTRL:-30}
cd /Users/ulysse/__projects__
PY=/Users/ulysse/__projects__/LLM_Blink/.venv/bin/python3
export PATH="/opt/homebrew/bin:$PATH"
R=LLM_Blink/results
LOADS="trivial math_bench_2 math_bench_4 math_bench_5"

for C in $CONDS; do
  case $C in
    rank1)                FLAGS=(--passphrase-rank 1); N=$N_MAIN ;;
    rank1_loadfirst)      FLAGS=(--passphrase-rank 1 --compute-order load_first); N=$N_MAIN ;;
    digits)               FLAGS=(--target-kind digits); N=$N_MAIN ;;
    trivmath)             FLAGS=(--target-kind trivial_math); N=$N_MAIN ;;
    rank1_exPF)           FLAGS=(--passphrase-rank 1 --example-order passphrase_first); N=$N_CTRL ;;
    rank1_loadfirst_exPF) FLAGS=(--passphrase-rank 1 --compute-order load_first --example-order passphrase_first); N=$N_CTRL ;;
    *) echo "unknown condition $C"; exit 2 ;;
  esac
  OUT="$R/cond_${C}_${SLUG}.csv"
  echo "===== $C: $MODEL -> $OUT  START $(date) ====="
  $PY -B LLM_Blink/probe_pilot.py --model "$MODEL" \
    --loads ${=LOADS} --regimes cot direct \
    --n-tasks 5 --n-seeds $N \
    --resample-full 20 --report-forks 0 --access on \
    --base-temp 1.0 --budget none --max-new-tokens 8192 \
    --n-workers 16 --keep-logs ${FLAGS[@]} \
    --out "$OUT" >> "$R/conditions_${C}_${SLUG}.log" 2>&1
  [ -f "${OUT%.csv}.samples.jsonl" ] && gzip -f "${OUT%.csv}.samples.jsonl"
  echo "===== $C: $MODEL  DONE $(date) ====="
done
