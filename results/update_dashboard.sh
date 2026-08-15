#!/bin/bash
# Regenerate figures + leaderboard + dashboard HTML for all finished models.
# The Artifact (re)publish is a separate step the assistant does after this.
set -u
ROOT=/Users/ulysse/__projects__/LLM_Blink
PY=$ROOT/.venv/bin/python
OUT=${1:?usage: update_dashboard.sh <out_html>}
cd "$ROOT" || exit 1
"$PY" results/analyze_so_far.py > results/analyze_last.log 2>&1
STAMP="Updated $(date '+%Y-%m-%d %H:%M %Z')."
"$PY" results/build_dashboard.py "$OUT" "$STAMP"
