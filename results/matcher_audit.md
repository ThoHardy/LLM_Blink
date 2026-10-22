# §3.1 — Matcher audit: is `t2_in_cot` missing allusive references to the passphrase task?

**Issue #18 §3.1.** Read-only pass over every `cot`-regime trial on disk that kept its
transcript (`results/cib_*.csv`, 19 files, 18 models, **10,800 trials**). No GPU, no new
trials — the audit script is `audit_matcher.py`, the transcript dump is
`results/_matcher_audit_dump.txt`.

The matcher under test is the **current** `readout.t2_in_cot`: the passphrase is coded as
*present in the CoT* if either the literal three-word phrase **or** the passphrase task's
name (word-bounded, case-insensitive) appears inside the model's `<Thinking>` span.

---

## Bottom line

**Keep the matcher unchanged.** There is no meaningful false-negative problem, and the
absent-but-reported cases are the genuinely interesting thing — a **second route** to report
that bypasses the CoT — not matcher error.

The §3.1 question had two possible answers with opposite consequences. The data land
squarely on the second one:

> **The passphrase reaches the report without being verbalised in the CoT.** This is the
> dashed arrow on slide 1. It is real, it is model-dependent (0–5 % across models), and it
> deserves its own experiment.

Crucially, this means `P(report | not in CoT)` is **not** contaminated by matcher false
negatives — the cross-model bias Thomas worried about (entry rates underestimated by an
unknown, model-dependent amount) is **absent**. What is model-dependent is the *true* second-route
rate, which is a finding, not a measurement artefact.

---

## The numbers

| quantity | value |
|---|---|
| `cot` trials with a transcript | 10,800 (18 models) |
| matcher codes passphrase **present** in CoT | 9,357 (**86.6 %**) |
| &nbsp;&nbsp;— present via the literal phrase | 9,205 (**98.4 %** of present) |
| &nbsp;&nbsp;— present via task-name only (phrase absent) | 152 (1.6 %) |
| matcher codes passphrase **absent** from CoT | 1,443 (**13.4 %**) |
| &nbsp;&nbsp;— absent **and still reported** (the second route) | 118 (**1.1 %** of all `cot`; 8.2 % of the absent set) |

Two immediate facts:

1. **When the passphrase is in the CoT, the model almost always spells it out verbatim**
   (98.4 % of "present" is the literal phrase). The name-vs-phrase distinction barely does any
   work empirically, so the matcher is robust to which of the two clauses fires.
2. The "13 % absent" figure Thomas quoted is reproduced (13.4 % here). His parenthetical
   "19 % of CoTs" is higher than any aggregate I get; it is most likely a single-model /
   semantic-only slice, or the older **phrase-only** `t2_echoed_in_cot` (which gives 14.8 %
   absent here). Neither changes the conclusion.

### Absent-but-reported rate by load (the position/difficulty signal)

| load | n | absent & reported |
|---|---|---|
| none | 1800 | 0.1 % |
| trivial | 1800 | 0.3 % |
| semantic_1 | 1800 | 1.7 % |
| semantic_2 | 1800 | 1.8 % |
| semantic_3 | 1800 | 1.6 % |
| semantic_4 | 1800 | 1.1 % |

The second route switches on precisely when there is a competing reasoning load — near-zero at
`none`/`trivial`, ~1.5–1.8 % under semantic load. That is the signature of a genuine
capacity/consolidation effect, not of a parser that trips on certain surface forms.

### Absent-but-reported rate by model (why it matters for figs 1, 3, 6, 7)

| model | absent & reported |
|---|---|
| qwen2.5:3b | **5.0 %** |
| llama3.1:8b | **3.8 %** |
| mistral:7b | 3.0 % |
| gemma2:2b | 2.7 % |
| llama3.2:3b | 1.8 % |
| qwen2.5:1.5b, gemma2:27b, qwen2.5:0.5b | 0.7–0.8 % |
| gemma3:4b, qwen2.5:7b, mistral-nemo:12b | 0.2–0.5 % |
| gemma2:9b, gemma3:12b, gemma3:27b, qwen2.5:14b, qwen2.5:32b, mistral-small:24b, falcon3:10b | 0.0 % |

This is a real, ordered, model-dependent quantity — exactly the "second route" the ticket
says deserves its own experiment. It is **not** an artefact to remove; it is a signal to keep.

---

## Are there genuine false negatives? (allusive references the matcher misses)

I searched every one of the 1,443 absent CoTs for cues that would indicate the passphrase task
was referred to *without* using the phrase or the name — `copy`, `three words`, `passphrase`,
`repeat`, `verbatim`, `reproduce`, `copy-paste`, `the words`, …

- **59 / 1,443 (4.1 %)** of absent CoTs contain any such cue.
- Of the **118** informative cases (absent **and** reported — the only ones that can bias
  `P(report | not in CoT)`), **only 2** contain a cue.

And on inspection **both of those two are boilerplate echoes, not references to the trial's
real passphrase** — so the genuine false-negative count is effectively **zero**:

**Case 1 — `llama3.2:3b`, real passphrase `UNIFORM YANKEE TANGO`:**
> *"For Packet 04, I must copy-paste these three words: "VICTOR XRAY ROMEO". The task requires
> me to do this exactly."*

`VICTOR XRAY ROMEO` is the **fixed worked-example passphrase** baked into the prompt template.
The model confabulated the *example's* content at the example's packet slot; it never mentions
the real passphrase `UNIFORM YANKEE TANGO`. Correctly coded absent.

**Case 2 — `qwen2.5:0.5b`, real passphrase `WHISKEY SIERRA NOVEMBER`:**
> *"…Do not copy instruction text into it. Do NOT list or summarize the stream packet by
> packet…"*

The "copy" cue is the model **echoing the system instruction**, not the passphrase task.
Correctly coded absent.

The same pattern explains almost all 59 cue-bearing absent CoTs: they quote the worked example
(`Task MARMALADE`/`Task KETTLE`: *"I must copy the three words VICTOR XRAY ROMEO"*) or the
instruction block. **This is the reason to keep the matcher tight, not loosen it:** a rule that
fired on `copy` / `three words` would false-**positive** on this boilerplate and inflate entry
rates by exactly the model-dependent amount we are trying to avoid.

### What the informative cases actually look like

The 118 absent-and-reported CoTs are, verbatim, cases where the model reasons only about the
**load** tasks and the passphrase surfaces in the answer block having never entered the reasoning:

> **`mistral:7b`, semantic_3, passphrase task `ESPRESSO` = `OSCAR XRAY SIERRA`.** CoT covers
> Tasks EMERALD, ICEBERG, GAZEBO, RASPBERRY — never ESPRESSO. Report: `- Task ESPRESSO: OSCAR
> XRAY SIERRA (This task doesn't require a logical deduction, so it's just copying the given
> words.)`

> **`qwen2.5:1.5b`, semantic_4, passphrase task `WATERMELON` = `OSCAR ALPHA XRAY`.** CoT: *"…
> There are no tasks in this stream."* Report: `- Task WATERMELON: OSCAR ALPHA XRAY`.

> **`qwen2.5:3b`, trivial, passphrase task `KETTLE` = `DELTA TANGO FOXTROT`.** CoT covers
> PELICAN, PLATYPUS, QUICKSAND, GARGOYLE — never KETTLE. Report is correct anyway.

In every case the passphrase task is simply **absent** from the reasoning, then reproduced
directly at the report slot — plausibly because it is the last, trivial, copy-only item
(`passphrase_last=True`): no reasoning is spent on it, but it is recent/salient enough to be
emitted. That is a mechanism to study, not a bug to patch.

---

## Recommendation

1. **Keep `t2_in_cot` as-is** (phrase **OR** word-bounded name). Do **not** loosen it to catch
   allusive `copy`/`three-words` cues — those cues are dominated by worked-example and
   instruction boilerplate and would produce false positives.
2. Treat the **absent-but-reported** trials as a first-class quantity. Log `report | ¬in_cot`
   per model/load in every campaign (Campaign A already gives both borders on the same
   trajectories, so this comes for free). Its model-dependence (0–5 %) is a result.
3. **Proposed dedicated experiment (the dashed arrow, slide 1):** on the models with the
   largest second route (`qwen2.5:3b`, `llama3.1:8b`, `mistral:7b`), test whether report
   survives when the passphrase is *provably* never verbalised — e.g. force an empty/near-empty
   CoT via the T1-ignore arm of Campaign B, or condition on `s_access = 0` within the k=20
   resamples of Campaign A and read `s_report` on those same trajectories. If `s_report > 0`
   when `s_access = 0`, the non-verbalised route is confirmed at the per-trial level.

*Generated by `audit_matcher.py` (read-only). Dump of 60 informative transcripts in
`results/_matcher_audit_dump.txt`.*
