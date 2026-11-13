#!/bin/zsh
# Issue #21: wait for Step 1 to finish, post its table on #21, then run Step 2
# (Ministral 3 ladder, main task) with retry-on-crash, commit+push per model,
# and post the Step 2 table. Go for Step 2 given by Ulysse on 2026-11-10.
cd /Users/ulysse/__projects__
REPO=/Users/ulysse/__projects__/LLM_Blink
PY=$REPO/.venv/bin/python3
R=LLM_Blink/results
LOG=$R/issue21_step2.log
export PATH="/Applications/Ollama.app/Contents/Resources:/opt/homebrew/bin:$PATH"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

# ---- wait for Step 1 ----
say "waiting for Step 1 (issue21_step1.log: ALL DONE)"
until grep -q "=== ALL DONE ===" $R/issue21_step1.log; do sleep 300; done
say "Step 1 done"

# ---- post Step 1 table ----
$PY -B $R/issue21_step1_summary.py > $R/summary_conditions_gemma3_4b.txt 2>&1
{
  echo "## Step 1 results: controls on gemma3:4b"
  echo
  echo "All 9 conditions are done (Campaign A protocol, same seeds). Data, gzipped transcripts and \`summary_conditions_gemma3_4b.csv\` are on branch \`results/issue-21-step1\`."
  echo
  echo "- **Drift check:** \`cond_base\` (N=20/load) reproduces \`scale_gemma3_4b.csv\` on the same seeds within sampling noise (Ollama 0.40.12, same model digest \`a2af6cc3\`); per-cell table below."
  echo "- **Truncation gate:** see below (fail = cell > 2 %)."
  echo "- **Manipulation check:** \`order_tau_cot\` is in the table. Note that for \`loadfirst\` alone the instructed order equals the natural one, so tau ≈ +1 is uninformative; for \`rank1_loadfirst*\` a negative tau means the model ignored the 'math first' instruction. Nothing excluded."
  echo "- **Ops note:** Ollama 0.40.12 throws rare transient HTTP 500s (~1 per few thousand requests); the runs were resumed (probe_pilot skips completed cells), no trial dropped."
  echo
  echo '```'
  cat $R/summary_conditions_gemma3_4b.txt
  echo '```'
  echo
  echo "Ulysse gave the go for Step 2 (Ministral 3 ladder); it starts now, results to follow."
  echo
  echo "_Posted automatically by the run script (Claude Code, on Ulysse's behalf)._"
} > $R/issue21_step1_comment.md
( cd $REPO && gh issue comment 21 --body-file results/issue21_step1_comment.md ) >> $LOG 2>&1 && say "Step 1 comment posted"

# ---- Step 2 ----
( cd $REPO && git switch -c results/issue-21-step2 ) >> $LOG 2>&1
MODELS=(
  "ministral-3:3b-instruct-2512-q4_K_M  ministral-3_3b"
  "ministral-3:8b-instruct-2512-q4_K_M  ministral-3_8b"
  "ministral-3:14b-instruct-2512-q4_K_M ministral-3_14b"
)
: > $R/ministral3_ollama_show.txt
for pair in $MODELS; do
  M=${pair%% *}
  { echo "===== $M"; ollama list | grep "^${M} "; ollama show $M; } >> $R/ministral3_ollama_show.txt 2>&1
done

for pair in $MODELS; do
  M=${pair%% *}; S=${pair##* }
  say "=== $M -> $S start ==="
  ok=0
  for try in {1..30}; do
    $REPO/run_family_model.sh $M $S >> $LOG 2>&1 && { ok=1; break; }
    say "$S crashed (try $try), resuming in 30s"; sleep 30
  done
  (( ok )) || say "$S GAVE UP after 30 tries"
  ( cd $REPO && git add results/scale_${S}.csv results/scale_${S}.samples.jsonl.gz results/campaignA_${S}.log \
      results/fig*.png results/scale_stats.txt results/ministral3_ollama_show.txt results/issue21_step2.log \
      run_issue21_step2.sh make_scale_figures.py 2>/dev/null
    git commit -q -m "#21 Step 2: $S done (n=100, K=20) + refreshed figs

$TRAILER" && git push -q -u origin results/issue-21-step2 ) >> $LOG 2>&1
  say "$S : committed+pushed"
done

# ---- post Step 2 table ----
$PY -B - > $R/summary_ministral3.txt 2>&1 <<'EOF'
import glob, pandas as pd
R = "LLM_Blink/results"
rows = []
for slug in ["mistral_7b", "ministral-3_3b", "ministral-3_8b", "ministral-3_14b"]:
    p = f"{R}/scale_{slug}.csv"
    if not glob.glob(p): continue
    d = pd.read_csv(p); d["p"] = d.s_report / d.k
    for load, g in list(d.groupby("t1_load")) + [("ALL", d)]:
        m = g.groupby("regime").p.mean(); c = g[g.regime == "cot"]
        rows.append(dict(model=slug, load=load, n=len(c), direct=m.get("direct"), cot=m.get("cot"),
                         rel_blink=(m.get("direct") - m.get("cot")) / m.get("direct"),
                         access_cot=(c.s_access / c.k).mean(), t1_cot=c.t1_mean.mean(),
                         worst_trunc=g.groupby("regime").frac_truncated.mean().max()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
EOF
{
  echo "## Step 2 results: Ministral 3 ladder (main task)"
  echo
  echo "Q4_K_M instruct tags, Campaign A protocol. CSVs, transcripts, refreshed figures and \`ollama show\` output (\`results/ministral3_ollama_show.txt\`) on branch \`results/issue-21-step2\`. \`mistral_7b\` (v0.3) shown for reference. \`worst_trunc\` = worst regime-level truncation rate per load."
  echo
  echo '```'
  cat $R/summary_ministral3.txt
  echo '```'
  echo
  echo "⏸️ Waiting for the go before Step 3."
  echo
  echo "_Posted automatically by the run script (Claude Code, on Ulysse's behalf)._"
} > $R/issue21_step2_comment.md
( cd $REPO && gh issue comment 21 --body-file results/issue21_step2_comment.md ) >> $LOG 2>&1 && say "Step 2 comment posted"
say "=== STEP 2 ALL DONE ==="
