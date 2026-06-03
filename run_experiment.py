"""Command-line entry point for the LLM Attentional Blink experiment.

Usage
-----
    # Ollama (local) -- any tag without a "/" is auto-detected as Ollama:
    python run_experiment.py --model gemma2:2b
    python run_experiment.py --model mistral:7b
    python run_experiment.py --model gemma3:12b --n-seeds 20

    # HuggingFace (Colab / GPU) -- any tag containing "/" is auto-detected as HF:
    python run_experiment.py --model Qwen/Qwen2.5-3B-Instruct
    python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit

The model name is the only required argument; everything else has sensible defaults.
Results are saved to <output> (default: ab_results_<model_slug>.csv).
"""

import argparse
import sys
import os

# Allow running from inside LLM_Blink/ or from the repo root (LLM_AB/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM_Blink import load_model, run_sweep, plot_ab  # noqa: E402


def _slug(model_name: str) -> str:
    """Turn a model name into a safe filename fragment."""
    return model_name.replace("/", "_").replace(":", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM Attentional Blink lag sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- model -----------------------------------------------------------------
    parser.add_argument(
        "--model", required=True,
        help=(
            "Model to use.  Ollama tag (no '/') for local use, e.g. 'gemma2:2b', "
            "'mistral:7b', 'gemma3:12b'.  HuggingFace Hub ID (contains '/') for "
            "Colab/GPU, e.g. 'Qwen/Qwen2.5-3B-Instruct'."
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

    # -- sweep parameters ------------------------------------------------------
    parser.add_argument(
        "--lags", nargs="+", type=int, default=[0, 1, 2, 3, 5, 8],
        metavar="LAG",
        help="Lag values to sweep (packets between T1 and T2).",
    )
    parser.add_argument(
        "--loads", nargs="+", default=["none", "easy", "hard"],
        choices=["none", "easy", "hard", "easy_math", "hard_math"],
        metavar="LOAD",
        help="T1 load levels to include.",
    )
    parser.add_argument(
        "--regimes", nargs="+", default=["cot"],
        choices=["cot", "direct"],
        metavar="REGIME",
        help="Answer regimes: 'cot' (chain-of-thought) and/or 'direct'.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of random seeds (trials) per condition cell.",
    )
    parser.add_argument(
        "--no-generation", action="store_true",
        help="Skip greedy decoding; only compute log-prob scores (faster).",
    )

    # -- output ----------------------------------------------------------------
    parser.add_argument(
        "--output", default=None,
        help="CSV output path. Defaults to ab_results_<model_slug>.csv.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show the AB curve plot after the sweep (requires a display).",
    )

    args = parser.parse_args()

    out_path = args.output or f"ab_results_{_slug(args.model)}.csv"

    print(f"Model  : {args.model}")
    print(f"Lags   : {args.lags}")
    print(f"Loads  : {args.loads}")
    print(f"Regimes: {args.regimes}")
    print(f"Seeds  : {args.n_seeds} per cell")
    print(f"Output : {out_path}")
    print()

    # -- load model ------------------------------------------------------------
    print("Loading model...")
    model, tok = load_model(
        args.model,
        load_in_4bit=args.load_in_4bit,
        ollama_base_url=args.ollama_url,
    )
    print("Model ready.\n")

    # -- sweep -----------------------------------------------------------------
    df = run_sweep(
        model, tok,
        lags=tuple(args.lags),
        loads=tuple(args.loads),
        regimes=tuple(args.regimes),
        n_seeds=args.n_seeds,
        do_generation=not args.no_generation,
        verbose=True,
    )

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df.groupby(["t1_load", "regime"])["t2_mean_logprob"].mean().to_string())

    # -- optional plot ---------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_ab(df, "t2_mean_logprob", ax=axes[0])
        if "report_correct" in df.columns:
            plot_ab(df, "report_correct", ax=axes[1])
        plt.suptitle(args.model, fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
