"""Re-score the graded T2 read-out of a saved results CSV with the exact
HuggingFace teacher-forced scorer.

Why this exists (2026-07-20): the Ollama backend cannot teacher-force, so its
"graded" numbers were pseudo-log-probs (generate-and-match with a -25 penalty
floor — see model.py::OllamaBackend.sequence_logprob). Runs generated on
Ollama are still fine on the REPORT side; this script recomputes the graded
side exactly, by rebuilding each trial's prompt deterministically from the
logged config/seed and scoring P(correct T2 | model's own generated prefix)
with a HF model (use the HF version of the same weights, e.g. gemma3:4b ->
google/gemma-3-4b-it; note Ollama default quantisation is Q4, HF fp16 — an
approximation, but a principled one).

Usage
-----
    # full re-scoring (needs GPU/CPU inference):
    python rescore_graded.py ab_results_gemma3_4b.csv --model google/gemma-3-4b-it

    # no model needed — just verify trials can be rebuilt bit-exact, and
    # (re)compute the t2_echoed_in_cot flag for old CSVs:
    python rescore_graded.py ab_results_gemma3_4b.csv --verify-only

Adds columns: rescore_ok, t2_echoed_in_cot, and (unless --verify-only)
t2_total_logprob_hf, t2_mean_logprob_hf, t2_joint_prob_hf, t2_n_tokens_hf.
Gate re-scored analyses on rescore_ok == True.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from LLM_Blink.stimuli import TrialConfig, build_trial  # noqa: E402
from LLM_Blink.experiment import _t2_scoring_prefix  # noqa: E402
from LLM_Blink.readout import t2_scoring_prefix_tasks  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def _as_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() == "true"
    return bool(x)


def _is_task_row(row) -> bool:
    """Combined-design (A x B x H) row? (n_tasks column present and set)."""
    return "n_tasks" in row.index and pd.notna(row["n_tasks"])


def _rebuild(row):
    """Rebuild the trial from the logged config/seed.

    Task-design rows (2026-07-20) rebuild from n_tasks/naming/passphrase_last
    directly (their geometry axes are ignored). Legacy rows: the original run
    may have drawn n_post randomly (default), or fixed n_post / n_pre via CLI
    flags; fixing them changes downstream RNG draws, so we try the candidate
    configs in order and accept the first whose realized stream matches the
    logged t2_phrase / n_pre / n_post exactly.
    """
    if _is_task_row(row):
        try:
            # anti_enumeration (2026-07-22): CSVs predating the column get
            # False -> byte-exact old prompt (the flag changes prompt text
            # only, never RNG, but the graded score conditions on the prompt).
            anti = (_as_bool(row["anti_enumeration"])
                    if "anti_enumeration" in row.index
                    and pd.notna(row["anti_enumeration"]) else False)
            # report_order (2026-08-15, Step 4): CSVs predating the column get
            # "none" -> byte-exact old prompt (same pattern as anti_enumeration).
            r_order = (str(row["report_order"])
                       if "report_order" in row.index
                       and pd.notna(row["report_order"]) else "none")
            # load_engagement (2026-08-15, Step 5): default "solve" for
            # pre-column CSVs -> byte-exact old prompt.
            l_eng = (str(row["load_engagement"])
                     if "load_engagement" in row.index
                     and pd.notna(row["load_engagement"]) else "solve")
            tr = build_trial(TrialConfig(
                t1_load=str(row["t1_load"]), regime=str(row["regime"]),
                t2_words=int(row["t2_words"]), seed=int(row["seed"]),
                n_tasks=int(row["n_tasks"]), naming=str(row["naming"]),
                passphrase_last=_as_bool(row["passphrase_last"]),
                anti_enumeration=anti, report_order=r_order,
                load_engagement=l_eng,
            ))
        except (ValueError, TypeError):
            return None
        if (tr.t2_phrase == str(row["t2_phrase"])
                and str(tr.t2_task_name) == str(row["t2_task_name"])
                and tr.t2_abs_index == int(row["t2_abs_index"])):
            return tr
        return None
    base = dict(lag=int(row["lag"]), t1_load=str(row["t1_load"]),
                regime=str(row["regime"]), mask=_as_bool(row["mask"]),
                t2_words=int(row["t2_words"]), seed=int(row["seed"]))
    candidates = (
        {},
        {"n_post": int(row["n_post"])},
        {"n_pre": int(row["n_pre"]), "n_post": int(row["n_post"])},
    )
    for extra in candidates:
        try:
            tr = build_trial(TrialConfig(**base, **extra))
        except (ValueError, TypeError):
            continue
        if (tr.t2_phrase == str(row["t2_phrase"])
                and tr.n_pre_used == int(row["n_pre"])
                and tr.n_post_used == int(row["n_post"])):
            return tr
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Re-score graded T2 log-probs of a saved CSV via HF "
                    "teacher forcing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("csv", help="results CSV produced by run_experiment.py")
    ap.add_argument("--model", default=None,
                    help="HF model ID to score with (the HF version of the "
                         "weights that generated the CSV). Required unless "
                         "--verify-only.")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <input>_rescored.csv)")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip scoring; only verify trial reconstruction and "
                         "(re)compute t2_echoed_in_cot.")
    args = ap.parse_args()

    if not args.verify_only and not args.model:
        ap.error("--model is required unless --verify-only is given.")

    df = pd.read_csv(args.csv)
    out_path = args.out or args.csv.replace(
        ".csv", "_verified.csv" if args.verify_only else "_rescored.csv")

    model = tok = None
    if not args.verify_only:
        from LLM_Blink import load_model
        from LLM_Blink.model import sequence_logprob
        if "/" not in args.model:
            ap.error("--model must be a HuggingFace ID (containing '/'); "
                     "scoring through Ollama is exactly what this script "
                     "works around.")
        print(f"Loading {args.model} ...")
        model, tok = load_model(args.model, load_in_4bit=args.load_in_4bit)

    nan = float("nan")
    ok_flags, echoes = [], []
    totals, means, joints, ntoks = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tr = _rebuild(row)
        raw = row["raw_output"]
        raw = "" if (isinstance(raw, float) and math.isnan(raw)) else str(raw)
        ok = tr is not None and raw != ""
        ok_flags.append(bool(ok))
        if not ok:
            echoes.append(None)
            totals.append(nan); means.append(nan); joints.append(nan)
            ntoks.append(0)
            continue
        if _is_task_row(row):
            prefix, _slot_missing = t2_scoring_prefix_tasks(
                raw, str(row["t2_task_name"]))
        else:
            prefix, _slot_missing = _t2_scoring_prefix(raw)
        if str(row["regime"]) == "cot":
            echoes.append(tr.t2_phrase in prefix.upper())
        else:
            echoes.append(None)
        if args.verify_only:
            totals.append(nan); means.append(nan); joints.append(nan)
            ntoks.append(0)
            continue
        lp = sequence_logprob(model, tok, system=tr.system,
                              user_prefix=tr.user, target_text=tr.t2_phrase,
                              prefilled_assistant=prefix)
        totals.append(lp["total_logprob"]); means.append(lp["mean_logprob"])
        joints.append(lp["joint_prob"]); ntoks.append(lp["n_tokens"])

    df["rescore_ok"] = ok_flags
    df["t2_echoed_in_cot"] = echoes
    if not args.verify_only:
        df["t2_total_logprob_hf"] = totals
        df["t2_mean_logprob_hf"] = means
        df["t2_joint_prob_hf"] = joints
        df["t2_n_tokens_hf"] = ntoks

    df.to_csv(out_path, index=False)
    n_ok = sum(ok_flags)
    print(f"\n{n_ok}/{len(df)} trials rebuilt bit-exact "
          f"({n_ok / max(len(df), 1):.1%}). Saved to {out_path}")
    if n_ok < len(df):
        print("Rows with rescore_ok=False could not be reconstructed "
              "(non-default sweep flags or missing raw_output) — gate on "
              "rescore_ok in analyses.")


if __name__ == "__main__":
    main()
