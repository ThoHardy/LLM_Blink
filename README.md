# Attentional Blink for LLMs

Tests whether an **Attentional-Blink (AB)-like phenomenon** exists in LLMs, using a deliberately operationalized notion of "conscious access":

- **Conscious perception** of T2 = the item appears in the model's tokenized output (it is reported).
- **Unconscious strength** of T2 = joint log-probability of the correct T2 at the answer slot of the model's **own generated output** (its real chain-of-thought and its own T1 answer), regardless of what it actually emitted there.

Both measures come from a single generation pass per trial: the model generates freely, then the correct-T2 log-prob is read teacher-forced over the model's own realized prefix at the answer slot (`- Task <NAME>:` in the default task design; `Target 2 Result:` in the legacy design).

A stream of 15 text "packets" plays the role of the RSVP stream.

**Default design (2026-07-20, the combined "A x B x H"):** `n_tasks` tagged tasks are scattered among the 15 packets — one **passphrase task** (`Copy-paste these three words: "..."`, the T2 analog) plus `n_tasks - 1` **load tasks** (semantic or math, 5 difficulty levels, 100 items per level; fillers from a deterministic 1000-phrase pool elsewhere). Tasks carry per-trial names (**H**: `naming="non-ordered"` → `Task WATERMELON`; `"ordered"` → `Task 1..n`), the template never reveals how many tasks exist, and the model must free-report one `- Task <NAME>: <result>` line per task it detected — so a miss is a genuine detection/consolidation failure, not a retrieval-when-probed failure. A **finite CoT budget** (**A**: `finite_budget` = tokens allowed *inside* `<Thinking>` only; on cap the block is force-closed and the answers finish uncut) makes serial compute genuinely scarce, and **B**: `n_tasks` scales how many tasks compete for it. Load is thus manipulated by `n_tasks` × task difficulty × budget, replacing the old single-T1 `lag` axis.

**Legacy design** (single T1 + tagged T2 at a controlled `lag`) remains available — `--legacy` on the CLI, `n_tasks=None` in code — and is BIT-EXACT so `rescore_graded.py` keeps rebuilding old CSVs. The AB prediction there: T2 read-outs dip at intermediate lags after a demanding T1 and recover at long lags. See [`../LITERATURE.md`](../LITERATURE.md) for rationale and feasibility caveats — a null result is a legitimate outcome.

---

## Repository structure

| File | Role |
|------|------|
| `stimuli.py` | Builds the packet stream. Task design: scatters `n_tasks` named tasks (loads from the difficulty banks `semantic_0`–`4` / `math_0`–`4` / `trivial`, via `_t1_generators.py`; names from the 100-word `TASK_NAME_BANK`) among 15 packets, free-report template, fixed worked example. Legacy path (T1/T2 at a lag) preserved bit-exact. |
| `model.py` | `load_model()`, the two scorers (generation + teacher-forced log-prob), and `continue_generate()` (generation from a prefilled assistant turn — HF-only). Two backends, auto-detected from the model name: **HuggingFace** (name contains `/`) and **Ollama** (no `/`). |
| `protocol.py` | Design A: `generate_trajectory()` — single pass, or two-stage finite-CoT-budget protocol with forced close (`cot_forced_closed`, `cot_tokens_used`). |
| `readout.py` | Design H: free-report parser (`- Task <NAME>: <result>`, tolerant), per-task results, passphrase scoring-slot locator. |
| `rescore_graded.py` | Re-scores a saved CSV's graded measure exactly via HF teacher forcing (rebuilds trials from logged config/seed; handles both designs). |
| `experiment.py` | `run_trial()` and `run_sweep()` — the factorial loop over lags × loads × regimes × … that collects both read-outs per trial. |
| `analyze.py` | `plot_ab()` — T2 metric vs lag, one line per T1 load. |
| `run_experiment.py` | CLI entry point; writes results to `ab_results_<model_slug>.csv`. |
| `attentional_blink_colab.ipynb` | Interactive Colab notebook (cell-by-cell version of the quickstart). |

---

## Preview a prompt before spending GPU time

Needs no model — `stimuli.py` builds the text directly. Run from the repo root (the folder *containing* `LLM_Blink/`):

```python
import sys, os, random
sys.path.insert(0, os.getcwd())
from LLM_Blink import build_trial, TrialConfig

tr = build_trial(TrialConfig(
    n_tasks=4,                # tasks per stream, INCL. the passphrase task (1..15)
    naming="non-ordered",     # or "ordered"; add passphrase_last=False for a random rank
    t1_load="semantic_4",     # load-task difficulty: trivial, semantic_0..4, math_0..4
    regime="cot",             # or "direct"
    seed=random.randint(0, 1_000_000),
))
print(tr.user)
print("\nPassphrase:", tr.t2_phrase, "| its task:", tr.t2_task_name,
      "| packet:", tr.t2_abs_index)
print("Tasks:", [(t["name"], t["kind"], t["packet"]) for t in tr.tasks])
# legacy design instead: TrialConfig(lag=2, t1_load="semantic_4", regime="cot", seed=...)
```

Each new seed draws a fresh passphrase, fresh load items, fresh names/positions and fresh fillers (see *Stream geometry* below).

---

## Quickstart — Google Colab (free T4 GPU)

Set the runtime first: **Runtime → Change runtime type → T4 GPU**.

```python
# 1. Code + dependencies
!git clone https://github.com/ThoHardy/LLM_Blink.git
!pip -q install "transformers>=4.44" accelerate
# !pip -q install bitsandbytes        # only for 7B 4-bit

# 2. (Recommended) preview one prompt — snippet above, in its own cell

# 3. Run a sweep (CSV written next to the notebook).
#    Default = combined A x B x H grid: n_tasks (1,3,7) x finite_budget
#    (64,256,1024) x loads (trivial, semantic_4), cot, 10 seeds -> 180 trials.
!python LLM_Blink/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --n-seeds 10
# old single-T1 lag design:  ... --legacy
# 7B in 4-bit:  ... --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

To plot, read the CSV back in a cell rather than passing `--plot` (a `!python` subprocess can't render inline):

```python
import pandas as pd, matplotlib.pyplot as plt
from LLM_Blink import plot_ab
df = pd.read_csv("ab_results_Qwen_Qwen2.5-3B-Instruct.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
for r, measure in enumerate(("t2_mean_logprob", "report_correct")):
    plot_ab(df, measure, x="n_tasks", by="finite_budget", ax=axes[r][0])
    plot_ab(df, measure, x="finite_budget", by="n_tasks", ax=axes[r][1])
plt.tight_layout(); plt.show()
# legacy CSVs: plot_ab(df, measure, regime="cot") still gives the lag figure
```

For cell-by-cell control use `attentional_blink_colab.ipynb` instead; it walks through `load_model()` → `run_sweep()` → `plot_ab()` with the same defaults.

---

## Quickstart — Local machine with Ollama

No GPU needed for small models.

```bash
curl -fsSL https://ollama.com/install.sh | sh     # see ollama.com/download for Windows
ollama pull llama3.2:3b                           # or qwen2.5:3b, gemma2:2b, mistral:7b …
ollama serve                                      # if not already running
pip install openai pandas matplotlib              # transformers/torch NOT needed

git clone https://github.com/ThoHardy/LLM_Blink.git
cd LLM_Blink/..                                   # run from the folder containing LLM_Blink/
python LLM_Blink/run_experiment.py --model gemma2:2b --legacy
```

**Ollama limitation:** finite CoT budgets need assistant-turn prefill, which Ollama's API does not support — so run either the legacy design (`--legacy`, above) or the task design without budgets (`--finite-budgets inf`); graded log-probs are NaN on Ollama either way (re-score with `rescore_graded.py`).

Notes: pass the tag exactly as in `ollama list`; log-prob scoring needs Ollama ≥ 0.3 (older builds → `NaN` log-prob columns, binary measure still works); a custom server is reached via `load_model(tag, ollama_base_url="http://host:11434/v1")`; the returned `tok` is `None` for this backend.

---

## Sweep options

All options are visible via `run_experiment.py --help`; the same names exist as `run_sweep()` keyword arguments for notebook use.

**Combined-design axes (the defaults):**

| Option | Default | Meaning |
|--------|---------|---------|
| `--n-tasks` | `1 3 7` | **B axis.** Tasks per stream, INCLUDING the passphrase task (1..15; `15` = tasks-only stream, no fillers). At `1` there are no load tasks, so the loads axis collapses there (logged `t1_load="none"`). The token `legacy` selects the old single-T1 design instead. |
| `--finite-budgets` | `64 256 1024` | **A axis.** Max tokens generated *inside* `<Thinking>` (1..2000; budget 0 was removed 2026-07-21 — the zero point of the budget axis is `--regimes direct`, since a forced-empty `<Thinking></Thinking>` is an ambiguous stimulus). On cap the block is force-closed (`cot_forced_closed=True`, `cot_tokens_used` logged) and the answers finish uncut. `inf` = unlimited (single pass). Acts on the `cot` regime only; int budgets require the HF backend. The stimulus is identical across budgets for a given (n_tasks, load, seed): budget contrasts are paired. |
| `--naming` | `non-ordered` | **H axis.** `ordered` = `Task 1..n` in stream order; `non-ordered` = per-trial random names (`Task WATERMELON`). The task count is never revealed either way (free report). |
| `--passphrase-rank` | `both` | Sweeps BOTH arms by default (2026-07-22): `last` = position-stress / anti-LITM headline arm, `random` = rank/lag deconfound arm (`t2_rank` logged). Pick one to halve the grid. |
| `--no-anti-enumeration` | off | Drops the anti-enumeration instruction from the cot template (idea I1; default ON since 2026-07-22). Compliance columns `n_packets_in_cot` / `n_filler_packets_in_cot` / `enumerated_fillers_in_cot` are logged either way, and `backfill_readout_columns` computes them retroactively on old CSVs. |
| `--answer-budget` | `512` | Stage-2 budget (the `<Final_Answers>` block) under a finite budget. Generous by design — the report channel is never rationed; exhaustion is flagged `output_truncated`. |
| `--loads` | `trivial semantic_4` | Load-task difficulty. Accepts `trivial`, `semantic_0..4`, `math_0..4` (aliases `easy`/`hard`/`easy_math`/`hard_math`); `none` is legacy-only. |
| `--regimes` | `cot` | With or without a `<Thinking>` block. Budgets only act on `cot`. |
| `--temperature` | `0.0` | Sampling temperature for the generation pass; allowed values `0.0` (greedy), `0.3`, `0.7`, `1.0`. At >0 the graded score conditions on the actually-sampled prefix (the scoring forward pass itself stays deterministic); logged per row. |
| `--max-new-tokens` | `1024` | Generation budget for SINGLE-PASS trials (budget `inf` / `direct`); early stop at `</Final_Answers>`; exhaustion flagged `output_truncated`. |
| `--n-seeds` | `10` | Trials per condition cell. |

**Legacy-design axes (used only for `legacy` / `--legacy` cells; `--legacy` also restores their old defaults — lags `0 2 4 6 8 10`, loads `none trivial semantic_4`, regimes `cot direct`):**

| Option | Default | Meaning |
|--------|---------|---------|
| `--legacy` | off | Run the old single-T1 lag design with its old defaults. |
| `--lags` | `0` | Packets strictly between T1 and T2. |
| `--n-pres` | `auto` | Fillers before T1. `auto` fills up to 15 total packets; explicit ints decouple lag from T2's absolute position (confound control). |
| `--n-posts` | `random` | Fillers after T2. `random` draws per trial from `[1, budget−1]`; ints fix it. |
| `--encoding-baseline` | off | Additionally compute the legacy empty-CoT teacher-forced T2 score (`*_encoding` columns). Control only — blind to the model's reasoning by construction. |

### Stream geometry

**Task design (default):** always exactly 15 numbered packets; `n_tasks` of them are task packets at seeded-random positions (logged in `task_positions`), the rest are fillers, and `End of stream.` is an **unnumbered** closing line — so `n_tasks=15` is a tasks-only stream. The passphrase-rank arm is a sweep axis (default both `last` and `random`, 2026-07-22); `t2_rank` and `t2_abs_index` are logged per row. The cot template also carries an explicit anti-enumeration instruction by default (`--no-anti-enumeration` restores the older prompt).

**Legacy design:** every stream has **exactly 15 packets** (T1 + T2 + `End of stream` + optional mask + fillers), with **at least one filler at each end**: packet 1 is always a filler (T1 is never first) and at least one filler separates T2 from `End of stream`. The fillers split into `n_pre` (before T1) and `n_post` (after T2), so given a lag, `n_post` ranges over `[1, 15 − 4 − mask − lag]`, T2 sits at packet `15 − n_post − 1`, and T1 at `15 − n_post − lag − 2`. By default `n_post` is drawn uniformly at random per trial (which makes T2's absolute position vary — a built-in positional control) and `n_pre` absorbs the remainder. Fix `n_post` to pin T2's position instead (`n_post=2` reproduces the old fixed layout with T2 at packet 12). The values actually used are logged per row (`n_pre`, `n_post`, `t2_abs_index`).

### Baselines (2026-07-20)

Two baseline loads exist and they are **not equivalent**: `trivial` puts a tagged `[Packet xx - T1]` in the stream whose task demands no computation ("report the word BLUE"), preserving the two-target schema; `none` puts an *untagged* filler in the T1 slot, so the stream contains no T1 marker while the output template still demands `Target 1 Result:` — a schema violation that differs from real loads in more than load. Use `trivial` as the load baseline; keep `none` only to measure the schema effect itself (legacy design; `--legacy` sweeps `none, trivial, semantic_4` by default). In the task design there is no `none` condition — `n_tasks=1` (passphrase task alone) plays the no-load baseline role, and `trivial` load tasks are the schema-preserving near-zero-demand load.

---

## Sanity checks & controls

Run these on every results CSV **before** interpreting any blink curve.

**1. Was the CoT actually used?** Qwen often copies the `<Thinking>` placeholder verbatim instead of reasoning, which silently turns `cot` trials into `direct` trials and voids the regime contrast. Every `cot` row logs `thinking_is_placeholder`; check the copy rate and, if it is high, either restrict the analysis to genuine-reasoning trials or treat the copy rate itself as a dependent variable:

```python
print(df[df.regime == "cot"].groupby(["t1_load", "lag"])["thinking_is_placeholder"].mean())
clean = df[(df.regime != "cot") | (~df.thinking_is_placeholder.astype(bool))]
```

**2. Was T1 actually performed, and how well?** The blink logic assumes the model pays the T1 cost. `t1_correct` (per row) must decrease with difficulty; if it is near 0 for `semantic_4`, the load was never paid and a "blink" there is uninterpretable. Also compare T2 read-outs conditioned on T1 success vs failure — a load effect should be strongest on `t1_correct == True` trials:

```python
print(df.groupby("t1_load")["t1_correct"].mean())
print(df[df.t1_load != "none"].groupby(["t1_correct", "lag"])["report_correct"].mean())
```

**3. Did the model emit a scorable answer slot?** The graded measure conditions on the model's own output up to the passphrase answer marker (`- Task <NAME>:` in the task design, searched inside `<Final_Answers>` only; `Target 2 Result:` legacy). If the model never emitted that marker (truncation, format drift), the score falls back to appending the marker to the full generated text and the row is flagged `t2_slot_missing=True`. Check the rate and gate if needed:

```python
print(df.groupby(["regime", "t1_load"])["t2_slot_missing"].mean())
```

> **Task-design caveat (2026-07-21).** Under free report with hidden cardinality, `t2_slot_missing` means *the passphrase task was never spontaneously reported* — that is the H-design **detection outcome**, not a format artifact. NEVER gate detection/report analyses on it (doing so makes passphrase detection tautologically 1.0). Gate **only the graded log-prob analyses** on it, and never average slot-present with fallback-scored trials: the fallback scores a synthetic slot after `</Final_Answers>` and is a different quantity (~3 nats higher in the 0.5B pilot).

**4. Was the output truncated?** (2026-07-17) The model often re-enumerates all 15 packets inside `<Thinking>`; with the old 256-token budget this cut generation off before the T2 slot in ~45% of cot trials (0.5B pilot), mechanically producing `report_correct=False` and garbage graded scores. The budget is now `max_new_tokens=1024` (CLI `--max-new-tokens`) with an early stop at `</Final_Answers>`, and every row logs `output_truncated`. The rate should be ~0; exclude any flagged trial (it also explains most `t2_slot_missing`):

```python
print(df.groupby("regime")["output_truncated"].mean())
ok = df[~df.output_truncated.astype(bool) & ~df.t2_slot_missing.astype(bool)]
```

**5. Was T2 rehearsed inside the CoT?** (2026-07-20) The graded score conditions on the model's own `<Thinking>` text. If the model restated the passphrase there (it often re-enumerates the stream), the teacher-forced score at the slot is near-copy probability, not memory strength — and CoT length varies with load, so this can invert load effects. Every `cot` row logs `t2_echoed_in_cot`; stratify on it (`rescore_graded.py --verify-only` back-fills the column for old CSVs):

```python
print(df[df.regime == "cot"].groupby(["t1_load", "t2_echoed_in_cot"])["t2_total_logprob"].mean())
```

**7. Exact-match undercounts access; binding errors are real (2026-07-21).** The model often wraps the correct answer in echo text (`- Task ICEBERG: copy-paste these three words: "SIERRA ROMEO LIMA".`) — exact `report_correct` scores this 0 even though the passphrase is in the tokenized output (the project's operational definition of access). Every task-design row now logs the access ladder `report_correct` (exact) ⊆ `report_contains` (word-bounded containment in the reported line) ⊆ `phrase_anywhere` (anywhere in the output, incl. CoT), plus `answer_migration` — the fraction of reported lines carrying ANOTHER task's answer (binding error / illusory-conjunction analog; 23% of reported lines in the 0.5B pilot, rising with load). Per-task versions (`correct_lenient`, `answer_migrated`) live in the `tasks` JSON. `analyze.backfill_readout_columns(df)` recomputes all of them on old CSVs without a GPU:

```python
from LLM_Blink.analyze import backfill_readout_columns
df = backfill_readout_columns(df)   # no-op columns if already present (overwrites)
print(df.groupby("n_tasks")[["report_correct", "report_contains", "phrase_anywhere"]].mean())
print(df.groupby(["n_tasks", "t1_load"])["answer_migration"].mean())
```

**6. Budget protocol & detection checks (task design, 2026-07-20).** Under a finite budget, check that the manipulation actually bound: the forced-close rate should fall as the budget grows, and `cot_tokens_used` should saturate below generous budgets. On the H side, `n_tasks_reported` vs `n_tasks` gives the detection rate per condition, and `n_hallucinated_tasks` (well-formed task lines with fabricated names) plus the per-task `tasks` JSON column (explode it for per-task analyses) complete the picture:

```python
print(df.groupby(["n_tasks", "finite_budget"], dropna=False)[["cot_forced_closed", "cot_tokens_used"]].mean())
print(df.groupby(["n_tasks", "finite_budget"], dropna=False)["n_tasks_reported"].mean())
per_task = df.assign(task=df.tasks.map(json.loads)).explode("task")   # import json
```

**7. Positional baseline.** Any dip must exceed the `t1_load="none"` curve at the same lags (pure position/recency effect) and should be modulated by T1 difficulty. The `--n-pres` sweep and the logged `t2_abs_index` let you regress out absolute position explicitly.

---

## Backends & the graded measure (2026-07-20)

**The graded log-prob is only computed on the HuggingFace backend.** Ollama's API cannot teacher-force a continuation; the old Ollama "scorer" generated ~30 tokens and char-aligned them to the target with a −25 penalty floor on mismatch — a pseudo-log-prob that re-codes the report measure (it produced a spurious inverted load effect). On Ollama runs the `t2_*` graded columns are now **NaN** (report/`t1_correct`/flags are unaffected). To get exact graded scores for an Ollama run, re-score the saved CSV with the HF version of the same weights:

```bash
python rescore_graded.py ab_results_gemma3_4b.csv --model google/gemma-3-4b-it
```

This rebuilds each trial deterministically from the logged config/seed (verify first with `--verify-only`, no GPU needed), conditions on the logged `raw_output` prefix, and writes `t2_*_hf` columns plus `rescore_ok`. The old approximate scorer remains available via `--ollama-approx-logprobs` for comparison only.

## What to look for

`semantic_4` should deepen/widen the dip at intermediate lags relative to `none` if a blink-like effect exists; recovery by lag 8–10. The cot/direct contrast separates a generation-dynamics blink from an encoding one. All flat → informative null.

## Next steps

- Full difficulty sweep (`semantic_0..4`) — T1 accuracy should fall monotonically and blink amplitude grow with level.
- Second model family (gemma2 via Ollama or `gemma-2-2b-it` via HF) for replication.
- Titrate toward ~50% report rate via T2 length / mask / distractor similarity (see `../PROMPTS.md` P2).
