"""Build the static MATH-Algebra load bank for issue #14 Step 6.

Pages the HuggingFace datasets-server JSON `/rows` API for the Algebra config of
`EleutherAI/hendrycks_math` (a mirror of Hendrycks et al. 2021), extracts the
`\\boxed{}` final answer, keeps only problems with a SHORT, unambiguous integer
answer and a compact statement (so a one-line report entry is exactly scorable),
and buckets them by MATH difficulty Level 1..5. Writes a static JSON bank so the
experiment never needs network or the `datasets` library at run time.

Run once (network required): `python3 build_math_bench.py`. Output:
`results/math_bench_algebra.json`  ->  {"1": [[question, answer], ...], ...}.
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.request

DATASET = "EleutherAI/hendrycks_math"
CONFIG = "algebra"
SPLITS = ("train", "test")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "math_bench_algebra.json")
MAX_PROBLEM_CHARS = 320       # keep packets compact
TARGET_PER_LEVEL = 100        # match the semantic banks' 100-item pools
INSTRUCTION = " Reply with only the final integer answer."


def _boxed(solution: str):
    """Return the content of the LAST \\boxed{...} with balanced braces, or None."""
    key = r"\boxed{"
    i = solution.rfind(key)
    if i == -1:
        return None
    j = i + len(key)
    depth, buf = 1, []
    while j < len(solution) and depth:
        c = solution[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        j += 1
    return "".join(buf).strip() if depth == 0 else None


def _clean_int(boxed: str):
    """Normalise a boxed answer to a plain integer string, or None if not one.

    Strips $, \\!, \\,, spaces, wrapping \\text{}. Accepts optional leading sign
    and comma thousands separators (1,000 -> 1000). Rejects anything else
    (fractions, decimals, variables, sets) so scoring stays exact."""
    s = boxed.strip().strip("$").strip()
    s = re.sub(r"\\[!,;: ]", "", s)
    m = re.match(r"\\text\{([^}]*)\}$", s)
    if m:
        s = m.group(1).strip()
    s = s.replace(",", "").replace(" ", "")
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    return None


def _fetch(offset: int, length: int, split: str):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config={CONFIG}&split={split}&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    banks = {str(i): [] for i in range(1, 6)}
    seen = set()
    for split in SPLITS:
        first = _fetch(0, 1, split)
        total = first.get("num_rows_total", 0)
        print(f"[{split}] {total} rows")
        for off in range(0, total, 100):
            if all(len(v) >= TARGET_PER_LEVEL for v in banks.values()):
                break
            for attempt in range(4):
                try:
                    d = _fetch(off, 100, split)
                    break
                except Exception as e:                       # transient 5xx / rate
                    print(f"   retry {off} ({e})"); time.sleep(2 * (attempt + 1))
            else:
                continue
            for r in d["rows"]:
                row = r["row"]
                lvl = str(row.get("level", "")).replace("Level ", "").strip()
                if lvl not in banks:
                    continue
                prob = row["problem"].strip()
                if len(prob) > MAX_PROBLEM_CHARS or "\n" in prob.strip():
                    continue
                ans = _boxed(row.get("solution", "") or "")
                if ans is None:
                    continue
                ans = _clean_int(ans)
                if ans is None:
                    continue
                q = prob + INSTRUCTION
                if q in seen:
                    continue
                seen.add(q)
                if len(banks[lvl]) < TARGET_PER_LEVEL:
                    banks[lvl].append([q, ans])
            print(f"   {split} off={off}: " +
                  " ".join(f"L{k}={len(v)}" for k, v in banks.items()), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(banks, f, ensure_ascii=False, indent=0)
    print("\nfinal counts:", {k: len(v) for k, v in banks.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
