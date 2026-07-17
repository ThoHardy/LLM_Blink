#!/usr/bin/env bash
# Gemma sweep runner (issue #9).
#
#   ./run_gemma_sweep.sh smoke                 # n_seeds=10, every local gemma tag
#   ./run_gemma_sweep.sh full                  # n_seeds=1000
#   ./run_gemma_sweep.sh smoke gemma2:2b gemma3:4b   # explicit model list
#
# Models default to every `gemma*` tag in `ollama list`. Output goes to
# results/ab_<slug>_t0_s<seeds>.csv inside the repo; existing files are skipped
# (delete the CSV to re-run a model). Fixed meta-parameters per the issue:
# temperature 0, default lags/loads/regimes, max_new_tokens 1024 (default).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:?usage: run_gemma_sweep.sh smoke|full [model ...]}"
shift || true
case "$MODE" in
  smoke) SEEDS=10 ;;
  full)  SEEDS=1000 ;;
  *) echo "mode must be 'smoke' or 'full'" >&2; exit 1 ;;
esac

if [ "$#" -gt 0 ]; then
  MODELS="$*"
else
  MODELS="$(ollama list | awk 'NR>1 {print $1}' | grep -i '^gemma' || true)"
  if [ -z "$MODELS" ]; then
    echo "no gemma models found in 'ollama list' — pass tags explicitly" >&2
    exit 1
  fi
fi

mkdir -p "$REPO_DIR/results"
echo "mode=$MODE seeds=$SEEDS"
echo "models:" $MODELS
echo

for M in $MODELS; do
  SLUG="$(echo "$M" | tr '/:' '__')"
  OUT="$REPO_DIR/results/ab_${SLUG}_t0_s${SEEDS}.csv"
  if [ -f "$OUT" ]; then
    echo "=== skip $M ($OUT exists)"
    continue
  fi
  echo "=== $M -> $OUT"
  START=$(date +%s)
  python "$REPO_DIR/run_experiment.py" \
    --model "$M" \
    --n-seeds "$SEEDS" \
    --temperature 0.0 \
    --output "$OUT"
  echo "=== $M done in $(( $(date +%s) - START ))s"
done

echo
echo "All done. Leaderboard:"
echo "  python \"$REPO_DIR/leaderboard.py\" \"$REPO_DIR\"/results/ab_*_s${SEEDS}.csv --retention"
