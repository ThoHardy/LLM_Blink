#!/bin/zsh
# Issue #21 Step 1 (gemma3:4b controls), autonomous: drift check, then all
# conditions; after each condition: summary + commit + push the results branch.
# Resumable (probe_pilot skips completed cells). Run under screen + caffeinate.
cd /Users/ulysse/__projects__
REPO=/Users/ulysse/__projects__/LLM_Blink
PY=$REPO/.venv/bin/python3
R=LLM_Blink/results
LOG=$R/issue21_step1.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
publish(){
  $PY -B $R/issue21_step1_summary.py > $R/summary_conditions_gemma3_4b.txt 2>&1
  ( cd $REPO && git add results/cond_*_gemma3_4b.csv results/cond_*_gemma3_4b.samples.jsonl.gz \
      results/conditions_*_gemma3_4b.log results/summary_conditions_gemma3_4b.* results/issue21_step1.log \
      results/issue21_step1_summary.py run_issue21_step1.sh 2>/dev/null
    git commit -q -m "#21 Step 1: gemma3:4b condition $1 done + summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin results/issue-21-step1 ) >> $LOG 2>&1
  say "$1 : committed+pushed"
}

say "=== drift check: base N=20 ==="
N_MAIN=20 $REPO/run_conditions_model.sh gemma3:4b gemma3_4b base >> $LOG 2>&1 || say "base FAILED"
publish base
for C in cue rank1 loadfirst rank1_loadfirst digits trivmath rank1_exPF rank1_loadfirst_exPF; do
  say "=== $C start ==="
  $REPO/run_conditions_model.sh gemma3:4b gemma3_4b $C >> $LOG 2>&1 || say "$C FAILED"
  publish $C
done
say "=== ALL DONE ==="
