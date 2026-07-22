"""Command-line entry point for the LLM Attentional Blink experiment.

Usage
-----
    # DEFAULT = combined A x B x H sweep: task streams (n_tasks incl. the
    # passphrase task), finite CoT budgets (tokens inside <Thinking> only),
    # non-ordered task names, free report. Requires a HuggingFace model
    # (name contains "/"):
    python run_experiment.py --model Qwen/Qwen2.5-3B-Instruct

    # pilot-sized:
    python run_experiment.py --model Qwen/Qwen2.5-3B-Instruct \\
        --n-tasks 1 7 --finite-budgets 128 1024 --n-seeds 3

    # LEGACY single-T1 lag design (old defaults; Ollama models work here):
    python run_experiment.py --model gemma3:4b --legacy

Results are saved to <output> (default: ab_results_<model_slug>.csv).
"""

import argparse
import sys
import os

# Allow running from inside LLM_Blink/ or from the repo root (LLM_AB/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Blink import load_model, run_sweep, plot_ab  # noqa: E402

# New-design defaults (the A x B x H trimmed grid). --legacy switches any of
# these still at their default back to the old single-T1 sweep values.
_DEF_N_TASKS = ["1", "3", "7"]
_DEF_BUDGETS = ["64", "256", "1024"]
_DEF_LAGS = [0]
_DEF_LOADS = ["trivial", "semantic_4"]
_DEF_REGIMES = ["cot"]


def _slug(model_name: str) -> str:
    """Turn a model name into a safe filename fragment."""
    return model_name.replace("/", "_").replace(":", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM Attentional Blink sweep "
                    "(default: combined A x B x H task-stream design).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- model -----------------------------------------------------------------
    parser.add_argument(
        "--model", required=True,
        help=(
            "Model to use.  Ollama tag (no '/') for local use, e.g. 'gemma2:2b', "
            "'mistral:7b', 'gemma3:12b'.  HuggingFace Hub ID (contains '/') for "
            "Colab/GPU, e.g. 'Qwen/Qwen2.5-3B-Instruct'. Finite budgets need HF."
        ),
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="HuggingFace only: load in 4-bit quantisation (needs bitsandbytes).",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434/v1",
        help="Ollama only: base URL of the Ollama server.",
    )
    parser.add_argument(
        "--ollama-approx-logprobs", action="store_true",
        help=(
            "Ollama only: re-enable the OLD approximate graded scorer. "
            "WARNING: it is not teacher-forced (it re-codes the report "
            "measure, with a -25 penalty floor on mismatch) — graded columns "
            "are then pseudo-log-probs. Default: graded columns are NaN on "
            "Ollama; use a HF model or rescore_graded.py for real scores."
        ),
    )

    # -- combined-design axes (A x B x H) --------------------------------------
    parser.add_argument(
        "--n-tasks", nargs="+", default=list(_DEF_N_TASKS), metavar="N",
        help=(
            "Design B axis: tasks per stream, INCLUDING the passphrase task "
            "(1..15; 15 = tasks-only stream, no fillers). n_tasks=1 has no "
            "load tasks, so the loads axis collapses there. The token "
            "'legacy' selects the old single-T1 lag design instead."
        ),
    )
    parser.add_argument(
        "--finite-budgets", nargs="+", default=list(_DEF_BUDGETS),
        metavar="BUDGET",
        help=(
            "Design A axis: max tokens generated INSIDE <Thinking> (0..2000); "
            "on cap the block is force-closed and answers finish uncut "
            "(cot_forced_closed flag). 'inf' = unlimited (single pass). "
            "Applies to the cot regime only; ints require a HF model."
        ),
    )
    parser.add_argument(
        "--naming", choices=["ordered", "non-ordered"], default="non-ordered",
        help=(
            "Design H axis: task names. 'ordered' = Task 1..n in stream "
            "order; 'non-ordered' = per-trial random names (Task WATERMELON)."
        ),
    )
    parser.add_argument(
        "--passphrase-rank", choices=["last", "random", "both"], default="both",
        help="Position of the passphrase task among the tasks. 'both' (default "
             "since 2026-07-22) sweeps the two arms: last = position-stress/"
             "anti-LITM headline, random = rank/lag deconfound.",
    )
    parser.add_argument(
        "--no-anti-enumeration", action="store_true",
        help="Drop the anti-enumeration instruction from the cot template "
             "(restores the pre-2026-07-22 prompt; compliance flags are "
             "logged either way).",
    )
    parser.add_argument(
        "--answer-budget", type=int, default=512,
        help=(
            "Stage-2 generation budget (the <Final_Answers> block) when a "
            "finite budget is applied. Generous by design: the report channel "
            "is never rationed; trials that exhaust it anyway are flagged "
            "output_truncated."
        ),
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help=(
            "Run the OLD single-T1 lag design with its old defaults "
            "(lags 0..10, loads none/trivial/semantic_4, cot+direct, no "
            "finite budget). Equivalent to --n-tasks legacy "
            "--finite-budgets inf --naming ordered plus old lag/load/regime "
            "defaults for any of those flags left unset."
        ),
    )

    # -- legacy sweep parameters -----------------------------------------------
    parser.add_argument(
        "--lags", nargs="+", type=int, default=list(_DEF_LAGS), metavar="LAG",
        help="LEGACY design only: lag values (packets between T1 and T2). "
             "Ignored for task-design cells.",
    )
    parser.add_argument(
        "--loads", nargs="+", default=list(_DEF_LOADS), metavar="LOAD",
        help=(
            "T1/load-task levels. Accepts 'none' (legacy only), 'trivial', "
            "the aliases 'easy'/'hard'/'easy_math'/'hard_math', or any "
            "explicit 'semantic_N' / 'math_N' (N=0..4)."
        ),
    )
    parser.add_argument(
        "--regimes", nargs="+", default=list(_DEF_REGIMES),
        choices=["cot", "direct"], metavar="REGIME",
        help="Answer regimes: 'cot' (chain-of-thought) and/or 'direct'. "
             "Finite budgets only act on 'cot'.",
    )
    parser.add_argument(
        "--n-pres", nargs="+", default=["auto"], metavar="NPRE",
        help="LEGACY design only: fillers before T1 ('auto' or ints).",
    )
    parser.add_argument(
        "--n-posts", nargs="+", default=["random"], metavar="NPOST",
        help="LEGACY design only: fillers after T2 ('random' or ints).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        choices=[0.0, 0.3, 0.7, 1.0],
        help=(
            "Sampling temperature for the generation (report) measure. "
            "0.0 = greedy decoding (default). Log-prob scoring is unaffected."
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help=(
            "Generation budget for SINGLE-PASS trials (finite budget 'inf' or "
            "direct regime). Generation stops early at </Final_Answers>; "
            "trials that exhaust it are flagged 'output_truncated'."
        ),
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of random seeds (trials) per condition cell.",
    )
    parser.add_argument(
        "--encoding-baseline", action="store_true",
        help=(
            "LEGACY design only: also compute the empty-CoT teacher-forced T2 "
            "score per trial ('*_encoding' columns)."
        ),
    )

    # -- output ----------------------------------------------------------------
    parser.add_argument(
        "--output", default=None,
        help="CSV output path. Defaults to ab_results_<model_slug>.csv.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show summary plots after the sweep (requires a display).",
    )

    args = parser.parse_args()

    # -- --legacy: restore old defaults for anything left unset ---------------
    if args.legacy:
        args.n_tasks = ["legacy"]
        args.finite_budgets = ["inf"]
        args.naming = "ordered"
        if args.lags == _DEF_LAGS:
            args.lags = [0, 2, 4, 6, 8, 10]
        if args.loads == _DEF_LOADS:
            args.loads = ["none", "trivial", "semantic_4"]
        if args.regimes == _DEF_REGIMES:
            args.regimes = ["cot", "direct"]

    # -- parse token lists -----------------------------------------------------
    n_tasks_list: list[int | None] = []
    for tk in args.n_tasks:
        if str(tk).lower() in ("legacy", "none"):
            n_tasks_list.append(None)
        else:
            v = int(tk)
            if not 1 <= v <= 15:
                parser.error(f"--n-tasks {v}: must be in 1..15 (or 'legacy').")
            n_tasks_list.append(v)

    finite_budgets: list[int | None] = []
    for tk in args.finite_budgets:
        if str(tk).lower() in ("inf", "none", "unlimited"):
            finite_budgets.append(None)
        else:
            v = int(tk)
            if not 0 < v <= 2000:
                parser.error(f"--finite-budgets {v}: must be in 1..2000 (or "
                             "'inf'). Budget 0 was removed 2026-07-21: the "
                             "zero point of the budget axis is the direct "
                             "regime (--regimes direct).")
            finite_budgets.append(v)

    if "/" not in args.model and any(b is not None for b in finite_budgets):
        parser.error(
            "finite budgets require a HuggingFace model (Ollama cannot "
            "continue a prefilled assistant turn). Use a HF ID, or "
            "--finite-budgets inf, or --legacy."
        )

    n_pres: list[int | None] = [None if t.lower() == "auto" else int(t)
                                for t in args.n_pres]
    n_posts: list[int | None] = [None if t.lower() == "random" else int(t)
                                 for t in args.n_posts]

    out_path = args.output or f"ab_results_{_slug(args.model)}.csv"

    print(f"Model    : {args.model}")
    print(f"n_tasks  : {n_tasks_list}")
    print(f"Budgets  : {finite_budgets} (CoT tokens; answer budget "
          f"{args.answer_budget})")
    print(f"Naming   : {args.naming}  (passphrase rank: {args.passphrase_rank}, "
          f"anti-enumeration: {not args.no_anti_enumeration})")
    print(f"Loads    : {args.loads}")
    print(f"Regimes  : {args.regimes}")
    if any(nt is None for nt in n_tasks_list):
        print(f"Lags     : {args.lags}  n_pres: {n_pres}  n_posts: {n_posts}")
    print(f"Temp.    : {args.temperature}")
    print(f"MaxTok   : {args.max_new_tokens} (single-pass trials)")
    print(f"Seeds    : {args.n_seeds} per cell")
    print(f"Output   : {out_path}")
    print()

    # -- load model ------------------------------------------------------------
    print("Loading model...")
    model, tok = load_model(
        args.model,
        load_in_4bit=args.load_in_4bit,
        ollama_base_url=args.ollama_url,
        ollama_approx_logprobs=args.ollama_approx_logprobs,
    )
    print("Model ready.\n")

    # -- sweep -----------------------------------------------------------------
    df = run_sweep(
        model, tok,
        naming=args.naming,
        n_tasks_list=tuple(n_tasks_list),
        finite_budgets=tuple(finite_budgets),
        loads=tuple(args.loads),
        regimes=tuple(args.regimes),
        passphrase_last=((True, False) if args.passphrase_rank == "both"
                         else args.passphrase_rank == "last"),
        anti_enumeration=not args.no_anti_enumeration,
        lags=tuple(args.lags),
        n_pres=tuple(n_pres),
        n_posts=tuple(n_posts),
        n_seeds=args.n_seeds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        answer_budget=args.answer_budget,
        encoding_baseline=args.encoding_baseline,
        verbose=True,
    )

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    if df["n_tasks"].notna().any():
        gb = ["t1_load", "n_tasks", "finite_budget"]
    else:
        gb = ["t1_load", "regime"]
    print(df.groupby(gb, dropna=False)[["report_correct", "t2_mean_logprob"]]
          .mean().to_string())

    # -- optional plot ---------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt
        if df["n_tasks"].notna().any():
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
            for r, measure in enumerate(("t2_mean_logprob", "report_correct")):
                plot_ab(df, measure, x="n_tasks", by="finite_budget",
                        ax=axes[r, 0])
                plot_ab(df, measure, x="finite_budget", by="n_tasks",
                        ax=axes[r, 1])
        else:
            regimes_present = sorted(df["regime"].unique())
            has_correct = "report_correct" in df.columns
            n_rows = 2 if has_correct else 1
            n_cols = len(regimes_present)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(6 * n_cols, 4 * n_rows),
                squeeze=False,
            )
            for col, regime in enumerate(regimes_present):
                plot_ab(df, "t2_mean_logprob", regime=regime, ax=axes[0, col])
                if has_correct:
                    plot_ab(df, "report_correct", regime=regime,
                            ax=axes[1, col])
        plt.suptitle(args.model, fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
