#!/bin/zsh
# Autonomous overnight orchestrator for issue #18 Campaign A (figs 2/3/6/7).
#
# 1. wait for the §3 pre-flight to finish (polls its log),
# 2. write the pre-flight report (§3.2/3.3/3.4 gate + titration),
# 3. run Campaign A on the Qwen2.5 ladder ascending, and after EACH model:
#    gzip its transcript log, regenerate the figures, and git-commit the CSV +
#    log + figures so the night's work is durable even across a reboot.
#
# Resumable: probe_pilot skips completed cells; re-launching continues. Runs
# unattended — commits carry the Claude co-author trailer.
cd /Users/ulysse/__projects__
PY=/Users/ulysse/__projects__/LLM_Blink/.venv/bin/python3
export PATH="/opt/homebrew/bin:$PATH"
R=LLM_Blink/results
LOG=$R/orchestrate.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

commit() {   # commit "<message>" file...
  local msg="$1"; shift
  git add "$@" 2>/dev/null
  git commit -q -m "$msg

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1 | tee -a $LOG
}

say "orchestrator start; waiting for pre-flight to finish"
while true; do
  grep -q "preflight ALL DONE" $R/preflight_run.log 2>/dev/null && { say "pre-flight ALL DONE"; break; }
  # robustness: if the pre-flight driver process is gone, proceed anyway — Campaign A
  # has its own per-cell truncation gate, and we must not hang the night on a partial
  # pre-flight (a model that died mid-scan still leaves usable titration for the rest).
  if ! pgrep -f preflight_driver.sh >/dev/null 2>&1; then
    say "pre-flight process gone (no ALL DONE) — proceeding; Campaign A self-gates truncation"
    break
  fi
  sleep 60
done

# §3 report
$PY -B LLM_Blink/preflight_analyze.py > $R/preflight_report.txt 2>&1 || true
say "wrote preflight_report.txt"
commit "#18 §3.2-3.4: pre-flight results — truncation gate, budget sizing, load titration" \
  $R/preflight_qwen2.5_0.5b.csv $R/preflight_qwen2.5_1.5b.csv \
  $R/preflight_qwen2.5_3b.csv $R/preflight_qwen2.5_7b.csv \
  $R/preflight_llama3.1_8b.csv $R/preflight_report.txt

LOADS="trivial math_bench_2 math_bench_4 math_bench_5"
for M in qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b; do
  SLUG=${M//:/_}
  say "Campaign A: $M START"
  $PY -B LLM_Blink/probe_pilot.py --model "$M" \
    --loads ${=LOADS} --regimes cot direct \
    --n-tasks 5 --n-seeds 100 \
    --resample-full 20 --report-forks 0 --access on \
    --base-temp 1.0 --budget none --max-new-tokens 8192 \
    --n-workers 8 --keep-logs \
    --out "$R/scale_${SLUG}.csv" >> $R/campaignA_${SLUG}.log 2>&1
  say "Campaign A: $M DONE"

  # compress the transcript log (text compresses ~6-10x) so it is committable
  [ -f "$R/scale_${SLUG}.samples.jsonl" ] && gzip -f "$R/scale_${SLUG}.samples.jsonl"
  $PY -B LLM_Blink/make_scale_figures.py $R/scale_*.csv >> $R/orchestrate.log 2>&1 || true
  commit "#18 Campaign A: ${M} done (n=100, K=20, both borders) + refreshed figs" \
    "$R/scale_${SLUG}.csv" "$R/scale_${SLUG}.samples.jsonl.gz" \
    "$R/scale_stats.txt" $R/fig2_slopegraph.png $R/fig3_blink_vs_size.png \
    $R/fig6_border_hists.png $R/fig7_leaderboards.png $R/campaignA_${SLUG}.log
done
say "Campaign A ladder (through 7b) COMPLETE"
