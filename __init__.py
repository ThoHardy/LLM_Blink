"""Attentional Blink for LLMs — experiment package.

Typical use (see ../README.md and the Colab notebook):

    from LLM_Blink.model import load_model
    from LLM_Blink.experiment import run_sweep
    from LLM_Blink.analyze import plot_ab

    model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
    df = run_sweep(model, tok, n_seeds=20)
    plot_ab(df, "t2_mean_logprob")
"""
from .stimuli import TrialConfig, build_trial, random_passphrase
from .model import load_model, report_generate, sequence_logprob
from .experiment import run_trial, run_sweep
from .analyze import summarize, plot_ab

__all__ = [
    "TrialConfig", "build_trial", "random_passphrase",
    "load_model", "report_generate", "sequence_logprob",
    "run_trial", "run_sweep", "summarize", "plot_ab",
]
