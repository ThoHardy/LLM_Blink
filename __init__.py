"""Attentional Blink for LLMs — experiment package.

Typical use (see ../README.md and the notebooks):

    from LLM_Blink.model import load_model
    from LLM_Blink.experiment import run_sweep
    from LLM_Blink.analyze import plot_ab

    model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
    df = run_sweep(model, tok, n_seeds=10)   # default = combined A x B x H grid
    plot_ab(df, "report_correct", x="finite_budget", by="n_tasks")
"""
from .stimuli import TrialConfig, build_trial, random_passphrase
from .model import (load_model, report_generate, continue_generate,
                    sequence_logprob)
from .protocol import generate_trajectory, Trajectory
from .experiment import run_trial, run_sweep
from .analyze import summarize, plot_ab, backfill_readout_columns

__all__ = [
    "TrialConfig", "build_trial", "random_passphrase",
    "load_model", "report_generate", "continue_generate", "sequence_logprob",
    "generate_trajectory", "Trajectory",
    "run_trial", "run_sweep", "summarize", "plot_ab",
    "backfill_readout_columns",
]
