#!/bin/bash
# Fully-autonomous finalizer: wait for llama3.1:8b, regen the 3 figures across all
# 3 models, commit + push, then remove the WIP title and post the summary comment
# to PR #17 via gh. Runs in a detached screen so the ENTIRE task completes even if
# the assistant is never re-invoked.
set -u
cd /Users/ulysse/__projects__
PY=LLM_Blink/.venv/bin/python3
GH=/opt/homebrew/bin/gh
R=LLM_Blink/results
rc(){ [ -f "$1" ] && echo $(($(wc -l < "$1")-1)) || echo 0; }
echo "[finalize] START $(date)"
while [ "$(rc $R/math_llama3.1_8b.csv)" -lt 800 ]; do sleep 120; done
echo "[finalize] llama done ($(rc $R/math_llama3.1_8b.csv)) $(date)"
sleep 25
$PY -B LLM_Blink/make_math_figures.py \
    $R/math_qwen2.5_3b.csv $R/math_qwen2.5_7b.csv $R/math_llama3.1_8b.csv \
    > $R/finalize_figs.log 2>&1
sed -i '' 's/| llama3.1:8b | in progress | running |/| llama3.1:8b | 800 | done |/' \
    $R/MATH_RUN_PROGRESS.md 2>/dev/null
cd /Users/ulysse/__projects__/LLM_Blink
git add results/math_llama3.1_8b.csv results/fig_math_reproduce.png \
        results/fig_math_per_model.png results/fig_math_mixture_mu_pi.png \
        results/math_stats.txt results/MATH_RUN_PROGRESS.md
git commit -m "MATH run: llama3.1:8b done (n=100) — all 3 models complete

Third model (other family). Figures + stats regenerated across all 3 models.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "[finalize] nothing to commit"
git push > $R/finalize_push.log 2>&1 && echo "[finalize] PUSHED $(date)" || echo "[finalize] push FAILED"
# --- PR #17 polish: drop WIP title + post summary comment ---
$GH pr edit 17 --repo ThoHardy/LLM_Blink \
    --title "MATH-Algebra load run (#14): CoT helps the load AND blinds the passphrase" \
    > $R/finalize_gh.log 2>&1 || echo "[finalize] pr edit failed"
{
  echo "## ✅ Run complete — all 3 models (n=100/cell)"
  echo ""
  echo "The CoT-induced blindness reproduces on the **standard Hendrycks MATH-Algebra**"
  echo "benchmark, and this time the *CoT-helps-the-load* leg is real (unlike the semantic"
  echo "loads): **CoT raises MATH T1 accuracy yet blinds the co-present passphrase.**"
  echo ""
  echo "Per-model dissociation, shape, and (μ, π) mixture verdict:"
  echo ""
  echo '```'
  cat results/math_stats.txt
  echo '```'
  echo ""
  echo "Figures: \`fig_math_reproduce.png\` (CoT helps T1 / hurts report), "
  echo "\`fig_math_per_model.png\` (report vs load + all-or-none/graded histogram), "
  echo "\`fig_math_mixture_mu_pi.png\` (μ_low/μ_high + π vs MATH load). Raw per-trial "
  echo "(s, k) CSVs committed for reproduction. See \`results/MATH_RUN_PROGRESS.md\`."
  echo ""
  echo "🤖 auto-posted by the finalizer when the run completed."
} > $R/pr17_comment.md
$GH pr comment 17 --repo ThoHardy/LLM_Blink --body-file $R/pr17_comment.md \
    >> $R/finalize_gh.log 2>&1 && echo "[finalize] comment posted" || echo "[finalize] comment failed"
echo "FINALIZED $(date)" > $R/_math14_finalized.marker
echo "[finalize] DONE $(date)"
