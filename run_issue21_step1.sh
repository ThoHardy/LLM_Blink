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
      results/issue21_step1_summary.py run_issue21_step1.sh model.py 2>/dev/null
    git commit -q -m "#21 Step 1: gemma3:4b condition $1 done + summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin results/issue-21-step1 ) >> $LOG 2>&1
  say "$1 : committed+pushed"
}

# Ollama 0.40.12 throws a rare transient HTTP 500 that kills probe_pilot; it is
# resumable, so relaunch the same condition until it exits cleanly.
run_cond(){  # $1 = condition, extra env passed through
  for try in {1..30}; do
    $REPO/run_conditions_model.sh gemma3:4b gemma3_4b $1 >> $LOG 2>&1 && return 0
    say "$1 crashed (try $try), resuming in 30s"; sleep 30
  done
  say "$1 GAVE UP after 30 tries"; return 1
}
# base (drift check) done 11-09; loadfirst done 11-10. 11-10 evening: model.py now
# maps Ollama's "token repeat limit" abort to a truncated sample (was a hard 500).
for C in trivmath cue rank1 rank1_loadfirst digits rank1_exPF rank1_loadfirst_exPF; do
  say "=== $C start ==="
  run_cond $C
  publish $C
done
say "=== ALL DONE ==="
