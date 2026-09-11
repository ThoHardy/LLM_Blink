"""Pre-flight analysis (issue #18 §3.2/§3.3/§3.4) over results/preflight_*.csv.

Reads the base-draw scans and reports, per model:
  §3.2 truncation rate per load x regime (gate: any cell > 2% is a fail at this
       max_new_tokens);
  §3.3 realised trajectory-length distribution (chars: p50/p95/p99/max) -> the
       budget the campaign should use (~2x p99 tokens, rounded to 4096/8192);
  §3.4 direct t1_correct per load -> titration: a rung is usable only if direct
       accuracy is off both floor and ceiling ([0.3, 0.7]); floor rungs make a
       FAKE blink and are flagged.

Chars->tokens uses ~3.6 chars/token (Qwen/Llama BPE, English+digits). Read-only.
Run from the folder CONTAINING LLM_Blink/.
"""
from __future__ import annotations
import csv, glob, os, sys, statistics as st

csv.field_size_limit(10 ** 7)
CHARS_PER_TOK = 3.6


def pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))
    return xs[i]


def main():
    paths = sorted(glob.glob("LLM_Blink/results/preflight_*.csv"))
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows:
            continue
        model = rows[0]["model"]
        loads = sorted({r["t1_load"] for r in rows})
        print(f"\n{'='*72}\n{model}   (n={len(rows)}, file={os.path.basename(path)})")

        # §3.3 budget: full-trajectory char length over ALL cot trials
        cot_lens = [int(r["cot_len_chars"]) for r in rows if r["regime"] == "cot"]
        if cot_lens:
            p50, p95, p99, mx = (pct(cot_lens, .5), pct(cot_lens, .95),
                                 pct(cot_lens, .99), max(cot_lens))
            tok99 = p99 / CHARS_PER_TOK
            rec = 8192 if 2 * tok99 > 4096 else 4096
            print(f"  §3.3 cot len chars: p50={p50} p95={p95} p99={p99} max={mx}"
                  f"  (~{tok99:.0f} tok p99) -> recommend --max-new-tokens {rec}")

        # §3.2 truncation + §3.4 titration, per load
        print("  §3.2/§3.4 per load:   trunc%(cot/dir)   direct_t1   cot_t1   verdict")
        for L in loads:
            cot = [r for r in rows if r["t1_load"] == L and r["regime"] == "cot"]
            dr = [r for r in rows if r["t1_load"] == L and r["regime"] == "direct"]
            def trunc(rs):
                return 100 * sum(int(r["base_output_truncated"]) for r in rs) / len(rs) if rs else 0.0
            def t1(rs):
                v = [float(r["t1_correct"]) for r in rs if r["t1_correct"] not in ("", None)]
                return sum(v) / len(v) if v else float("nan")
            dt1 = t1(dr)
            verdict = "usable"
            if dt1 == dt1:  # not nan
                if dt1 < 0.3:
                    verdict = "FLOOR (fake blink risk)"
                elif dt1 > 0.7:
                    verdict = "ceiling"
            tflag = ""
            if trunc(cot) > 2 or trunc(dr) > 2:
                tflag = "  <-- TRUNC>2% re-run bigger"
            print(f"    {L:16s}   {trunc(cot):4.1f}/{trunc(dr):4.1f}       "
                  f"{dt1:5.2f}       {t1(cot):5.2f}   {verdict}{tflag}")


if __name__ == "__main__":
    main()
