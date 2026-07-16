# Attentional Blink for LLMs

Tests whether an **Attentional-Blink (AB)-like phenomenon** exists in LLMs, using a deliberately operationalized notion of "conscious access":

- **Conscious perception** of T2 = the item appears in the model's tokenized output (it is reported).
- **Unconscious strength** of T2 = joint log-probability of the correct T2 at the answer slot of the model's **own generated output** (its real chain-of-thought and its own T1 answer), regardless of what it actually emitted there.

Both measures come from a single generation pass per trial: the model generates freely, then the correct-T2 log-prob is read teacher-forced over the model's own realized prefix at the `Target 2 Result:` slot.

A stream of 15 text "packets" plays the role of the RSVP stream: T1 is a capacity-demanding task (semantic or math, 5 difficulty levels, 100 items per level), T2 is a novel 3-word NATO passphrase placed `lag` packets after T1, and the remaining packets are fillers drawn from a deterministic 1000-phrase pool. The AB prediction: T2 read-outs dip at intermediate lags after a demanding T1 and recover at long lags. See [`../LITERATURE.md`](../LITERATURE.md) for rationale and feasibility caveats — a null result is a legitimate outcome.

---

## Repository structure

| File | Role |
|------|------|
| `stimuli.py` | Builds the packet stream: samples T1 from the difficulty banks (`semantic_0`–`4`, `math_0`–`4`; 100 items each, via `_t1_generators.py`), generates the T2 passphrase, inserts it at the given lag, fills the rest from the 1000-phrase `FILLER_POOL`, and prefixes a fixed worked example. |
| `model.py` | `load_model()` plus the two scorers (generation + teacher-forced log-prob). Two backends, auto-detected from the model name: **HuggingFace** (name contains `/`) and **Ollama** (no `/`). |
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
    lag=2,                    # 0, 2, 4, 6, 8, 10 are the sweep defaults
    t1_load="semantic_4",     # "none", semantic_0..4, math_0..4
    regime="cot",             # or "direct"
    seed=random.randint(0, 1_000_000),
))
print(tr.user)
print("\nT2 passphrase:", tr.t2_phrase, "| T1 answer:", tr.t1_answer,
      "| T2 packet index:", tr.t2_abs_index)
```

Each new seed draws a fresh T2, a fresh T1 item, fresh fillers, and a fresh number of post-T2 packets (see *Stream geometry* below).

---

## Quickstart — Google Colab (free T4 GPU)

Set the runtime first: **Runtime → Change runtime type → T4 GPU**.

```python
# 1. Code + dependencies
!git clone https://github.com/ThoHardy/LLM_Blink.git
!pip -q install "transformers>=4.44" accelerate
# !pip -q install bitsandbytes        # only for 7B 4-bit

# 2. (Recommended) preview one prompt — snippet above, in its own cell

# 3. Run a sweep (CSV written next to the notebook)
!python LLM_Blink/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --n-seeds 10
# 7B in 4-bit:  ... --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

To plot, read the CSV back in a cell rather than passing `--plot` (a `!python` subprocess can't render inline):

```python
import pandas as pd, matplotlib.pyplot as plt
from LLM_Blink import plot_ab
df = pd.read_csv("ab_results_Qwen_Qwen2.5-3B-Instruct.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
for col, regime in enumerate(sorted(df["regime"].unique())):
    plot_ab(df, "t2_mean_logprob", regime=regime, ax=axes[0][col])   # graded
    if "report_correct" in df.columns:
        plot_ab(df, "report_correct", regime=regime, ax=axes[1][col])  # binary
plt.tight_layout(); plt.show()
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
python LLM_Blink/run_experiment.py --model gemma2:2b
```

Notes: pass the tag exactly as in `ollama list`; log-prob scoring needs Ollama ≥ 0.3 (older builds → `NaN` log-prob columns, binary measure still works); a custom server is reached via `load_model(tag, ollama_base_url="http://host:11434/v1")`; the returned `tok` is `None` for this backend.

---

## Sweep options

All options are visible via `run_experiment.py --help`; the same names exist as `run_sweep()` keyword arguments for notebook use.

| Option | Default | Meaning |
|--------|---------|---------|
| `--lags` | `0 2 4 6 8 10` | Packets strictly between T1 and T2. |
| `--loads` | `none semantic_4` | T1 difficulty. Accepts `none`, `semantic_0..4`, `math_0..4` (aliases `easy`/`hard`/`easy_math`/`hard_math`). |
| `--regimes` | `cot direct` | With or without a `<Thinking>` block before the answers. |
| `--n-pres` | `auto` | Fillers before T1. `auto` fills up to 15 total packets; explicit ints (`--n-pres 2 4 6 8`) decouple lag from T2's absolute position (confound control). |
| `--n-posts` | `random` | Fillers after T2. `random` draws per trial from `[1, budget−1]`; ints fix it. See *Stream geometry*. |
| `--temperature` | `0.0` | Sampling temperature for the generation pass; allowed values `0.0` (greedy), `0.3`, `0.7`, `1.0`. At >0 the graded score conditions on the actually-sampled prefix (the scoring forward pass itself stays deterministic); `temperature` is logged per row. |
| `--n-seeds` | `10` | Trials per condition cell. |
| `--encoding-baseline` | off | Additionally compute the legacy empty-CoT teacher-forced T2 score (`*_encoding` columns): P(correct T2 | stream, empty reasoning, force-fed correct T1). Control only — blind to the model's reasoning by construction. |

### Stream geometry

Every stream has **exactly 15 packets** (T1 + T2 + `End of stream` + optional mask + fillers), with **at least one filler at each end**: packet 1 is always a filler (T1 is never first) and at least one filler separates T2 from `End of stream`. The fillers split into `n_pre` (before T1) and `n_post` (after T2), so given a lag, `n_post` ranges over `[1, 15 − 4 − mask − lag]`, T2 sits at packet `15 − n_post − 1`, and T1 at `15 − n_post − lag − 2`. By default `n_post` is drawn uniformly at random per trial (which makes T2's absolute position vary — a built-in positional control) and `n_pre` absorbs the remainder. Fix `n_post` to pin T2's position instead (`n_post=2` reproduces the old fixed layout with T2 at packet 12). The values actually used are logged per row (`n_pre`, `n_post`, `t2_abs_index`).

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

**3. Did the model emit a scorable T2 slot?** The graded measure conditions on the model's own output up to the `Target 2 Result:` marker. If the model never emitted that marker (truncation, format drift), the score falls back to appending the marker to the full generated text and the row is flagged `t2_slot_missing=True`. Check the rate and gate if needed:

```python
print(df.groupby(["regime", "t1_load"])["t2_slot_missing"].mean())
```

**4. Positional baseline.** Any dip must exceed the `t1_load="none"` curve at the same lags (pure position/recency effect) and should be modulated by T1 difficulty. The `--n-pres` sweep and the logged `t2_abs_index` let you regress out absolute position explicitly.

---

## What to look for

`semantic_4` should deepen/widen the dip at intermediate lags relative to `none` if a blink-like effect exists; recovery by lag 8–10. The cot/direct contrast separates a generation-dynamics blink from an encoding one. All flat → informative null.

## Next steps

- Full difficulty sweep (`semantic_0..4`) — T1 accuracy should fall monotonically and blink amplitude grow with level.
- Second model family (gemma2 via Ollama or `gemma-2-2b-it` via HF) for replication.
- Titrate toward ~50% report rate via T2 length / mask / distractor similarity (see `../PROMPTS.md` P2).
