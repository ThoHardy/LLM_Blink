"""Union-probability "virtual cued probe" for the graded read-out (idea I3).

Problem (2026-07-21/22 pilots): on 50-90% of cot trials the passphrase task is
never spontaneously reported, so there is no principled `Task <NAME>:` slot to
teacher-force the graded measure at (`t2_slot_missing`), and the fallback
scores a different quantity. This script measures graded access to the target
task's IDENTITY without needing its slot to exist: at EVERY task-name slot the
model actually emitted in <Final_Answers> (each "- Task " line), it computes
the teacher-forced probability that the TARGET's name would have been emitted
there instead, then combines across slots:

    P(target mentioned somewhere) ~ 1 - prod_i (1 - p_i)

Per slot it also reports the BANK-NORMALIZED relative probability
p_target / sum_{name in bank} p_name (default: the full 100-name
TASK_NAME_BANK; --norm-set trial restricts to the trial's own task names),
which removes the "how plausible is ANY name here" component.

Retroactive: works on any saved task-design CSV (rebuilds each trial
bit-exact from the logged config/seed via rescore_graded._rebuild, incl. the
anti_enumeration default-False rule for old CSVs). HF backend only.

NOT implemented (Thomas, 2026-07-22): the second stage
P(passphrase | slot with correct name forced) — deferred, the answer format
is not reliably respected.

Interpretation caveats: (a) slots where the model DID report the target name
score near 1 by construction — stratify on t2_name_reported (or
report_correct) like t2_echoed_in_cot; (b) candidates are scored in canonical
bank casing; a model that lowercases names is underestimated (the tolerant
parser is case-insensitive, the probe is not).

Usage
-----
    python union_probe.py ab_results_qwen1.5b_hab_cot_passphraselast.csv \
        --model Qwen/Qwen2.5-1.5B-Instruct

    # no GPU: rebuild check + slot discovery only
    python union_probe.py <csv> --verify-only

Adds columns: uprobe_ok, n_name_slots, t2_name_reported, and (unless
--verify-only) t2_name_logprobs, t2_name_relprobs (JSON lists, one per slot),
t2_name_prob_union, t2_name_prob_max, t2_name_relprob_union,
t2_name_relprob_max, uprobe_norm_set. Gate analyses on uprobe_ok == True.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from LLM_Blink.stimuli import TASK_NAME_BANK  # noqa: E402
from LLM_Blink.readout import _answers_region, _TASK_LINE_RE  # noqa: E402
from LLM_Blink.model import _build_prompt_ids  # noqa: E402
from rescore_graded import _rebuild, _is_task_row  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def find_name_slots(raw_output: str):
    """(prefix, leading_space) per emitted task line, in stream order.

    prefix = raw_output up to (excluding) the name, trailing spaces stripped;
    leading_space = whether a space separated prefix and name (so candidates
    are tokenized as " NAME", the natural BPE boundary).
    Only lines in the answers region (after <Final_Answers>) count — the CoT
    may legitimately restate tasks (same rule as the report parser).
    """
    region, off = _answers_region(raw_output)
    slots = []
    for m in _TASK_LINE_RE.finditer(region):
        cut = off + m.start(1)
        pre = raw_output[:cut]
        stripped = pre.rstrip(" ")
        slots.append((stripped, len(stripped) < len(pre)))
    return slots


def _score_slot(torch, F, model, tok, prompt_ids, prefix: str,
                leading_space: bool, candidates: list) -> dict:
    """log P(candidate | system+user+prefix) for every candidate name.

    One forward over the prefix (cached), then one tiny forward per
    multi-token candidate (single-token candidates read straight off the
    prefix logits). DynamicCache is cropped back after each candidate;
    legacy tuple caches need no crop (not mutated in place).
    """
    dev = model.device
    pre_ids = tok(prefix, add_special_tokens=False,
                  return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, pre_ids], dim=1).to(dev)
    n_prefix = ids.shape[1]
    out = model(ids, use_cache=True)
    past = out.past_key_values
    last = F.log_softmax(out.logits[0, -1].float(), dim=-1)

    sp = " " if leading_space else ""
    lps = {}
    for cand in candidates:
        ct = tok(sp + cand, add_special_tokens=False,
                 return_tensors="pt").input_ids[0].to(dev)
        lp = last[ct[0]].item()
        if ct.shape[0] > 1:
            inp = ct[:-1].unsqueeze(0)
            mask = torch.ones((1, n_prefix + inp.shape[1]), dtype=torch.long,
                              device=dev)
            o2 = model(inp, past_key_values=past, attention_mask=mask)
            if hasattr(past, "crop"):
                past.crop(n_prefix)
            lg = F.log_softmax(o2.logits[0].float(), dim=-1)
            for i in range(ct.shape[0] - 1):
                lp += lg[i, ct[i + 1]].item()
        lps[cand] = lp
    return lps


def _null_cols():
    return dict(uprobe_ok=False, n_name_slots=None, t2_name_reported=None,
                t2_name_logprobs=None, t2_name_relprobs=None,
                t2_name_prob_union=None, t2_name_prob_max=None,
                t2_name_relprob_union=None, t2_name_relprob_max=None)


def probe_row(row, model, tok, torch, F, norm_set: str,
              verify_only: bool) -> dict:
    if not _is_task_row(row):
        return _null_cols()
    tr = _rebuild(row)
    if tr is None:
        return _null_cols()
    raw = str(row.get("raw_output") or "")
    slots = find_name_slots(raw)
    target = str(tr.t2_task_name)
    reported = any(m.group(1).upper() == target.upper()
                   for m in _TASK_LINE_RE.finditer(_answers_region(raw)[0]))
    cols = _null_cols()
    cols.update(uprobe_ok=True, n_name_slots=len(slots),
                t2_name_reported=reported)
    if verify_only or not slots:
        return cols

    if norm_set == "trial":
        candidates = [t["name"] for t in tr.tasks]
    else:
        candidates = list(TASK_NAME_BANK)
    if target not in candidates:
        candidates.append(target)

    prompt_ids = _build_prompt_ids(tok, tr.system, tr.user,
                                   add_generation_prompt=True)
    t_lps, rel_ps = [], []
    with torch.no_grad():
        for prefix, sp in slots:
            lps = _score_slot(torch, F, model, tok, prompt_ids, prefix, sp,
                              candidates)
            lp_t = lps[target]
            denom = math.log(sum(math.exp(v - max(lps.values()))
                                 for v in lps.values())) + max(lps.values())
            t_lps.append(lp_t)
            rel_ps.append(math.exp(lp_t - denom))

    ps = [math.exp(v) for v in t_lps]
    cols.update(
        t2_name_logprobs=json.dumps(t_lps),
        t2_name_relprobs=json.dumps(rel_ps),
        t2_name_prob_union=1.0 - math.prod(1.0 - min(p, 1.0) for p in ps),
        t2_name_prob_max=max(ps),
        t2_name_relprob_union=1.0 - math.prod(1.0 - p for p in rel_ps),
        t2_name_relprob_max=max(rel_ps),
    )
    return cols


def main():
    ap = argparse.ArgumentParser(
        description="Union-probability virtual cued probe (idea I3) over a "
                    "saved task-design CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--model", default=None,
                    help="HF model id used for the original run (REQUIRED "
                         "unless --verify-only). Must be the HF backend.")
    ap.add_argument("--norm-set", choices=["bank", "trial"], default="bank",
                    help="Denominator of the relative probability: full "
                         "100-name TASK_NAME_BANK or the trial's own names.")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="No model: rebuild check + slot discovery only.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N rows (debug).")
    ap.add_argument("--output", default=None,
                    help="Output CSV (default: <input>_uprobe.csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[:args.limit].copy()

    model = tok = torch = F = None
    if not args.verify_only:
        if not args.model:
            ap.error("--model is required unless --verify-only")
        import torch  # noqa: F811
        import torch.nn.functional as F  # noqa: F811
        from LLM_Blink.model import load_model
        model, tok = load_model(args.model, load_in_4bit=args.load_in_4bit)

    new_cols = None
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cols = probe_row(row, model, tok, torch, F, args.norm_set,
                         args.verify_only)
        if new_cols is None:
            new_cols = {k: [] for k in cols}
            new_cols["uprobe_norm_set"] = []
        for k, v in cols.items():
            new_cols[k].append(v)
        new_cols["uprobe_norm_set"].append(
            None if args.verify_only else args.norm_set)
    for k, v in (new_cols or {}).items():
        df[k] = v

    out = args.output or args.csv.replace(".csv", "_uprobe.csv")
    df.to_csv(out, index=False)
    ok = df["uprobe_ok"].mean() if len(df) else float("nan")
    print(f"Saved {len(df)} rows to {out}  (uprobe_ok rate: {ok:.2f})")
    if not args.verify_only and "t2_name_relprob_union" in df:
        print(df.groupby(["t1_load", "finite_budget"], dropna=False)
              ["t2_name_relprob_union"].mean().round(3))


if __name__ == "__main__":
    main()
