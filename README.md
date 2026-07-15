# Attentional Blink for LLMs

Tests whether an **Attentional-Blink (AB)-like phenomenon** exists in LLMs, using a deliberately operationalized notion of "conscious access":

- **Conscious perception** of T2 = the item appears in the model's tokenized output (it is reported).
- **Unconscious strength** = joint log-probability of the correct T2 under the forced template, regardless of what was generated.

See [`../LITERATURE.md`](../LITERATURE.md) for the design rationale and feasibility caveats.

---

## Repository structure

| File | Role |
|------|------|
| `model.py` | Model loading and scoring. Supports two backends: **HuggingFace** (`transformers` + `torch`) for GPU-accelerated models, and **Ollama** (via OpenAI-compatible API) for local CPU inference. Exposes `load_model()`, a log-prob scorer, and a greedy-generation scorer. |
| `stimuli.py` | Stimulus generation. Builds the RSVP-like packet stream, samples T1 items from difficulty-graded banks (`semantic_0`–`4`, `math_0`–`4`), generates T2 passphrases, inserts them at a given lag, and draws filler packets from a **1000-phrase deterministic `FILLER_POOL`** (system-log style) seeded at module load. Also prefixes every prompt with a fixed worked example so the model sees the expected output form. |
| `experiment.py` | Trial logic. Defines `TrialConfig`, `build_trial()`, and `run_sweep()` — the main loop that iterates over lags, loads, and regimes and collects both read-outs (binary report + joint log-prob). |
| `analyze.py` | Analysis and plotting. `plot_ab()` draws the blink curve (T2 metric vs lag, one line per T1 load). Also contains aggregate helpers. |
| `run_experiment.py` | CLI entry point. Parses arguments (`--model`, `--lags`, `--loads`, `--regimes`, `--n-seeds`, `--plot`, …), runs a full sweep, and writes results to a CSV. |
| `__init__.py` | Package exports (`load_model`, `run_sweep`, `plot_ab`, `build_trial`, `TrialConfig`). |
| `attentional_blink_colab.ipynb` | Interactive Colab notebook for cell-by-cell exploration. |
| `pyproject.toml` | Package metadata — enables `pip install git+https://github.com/ThoHardy/LLM_Blink`. |
| `requirements_local.txt` | Local dev dependencies (for running outside Colab). |

---


## Quickstart — Google Colab (free T4 GPU)

First set the runtime: **Runtime → Change runtime type → T4 GPU**.

### Fastest path — clone the repo and run one command

The same CLI used for Ollama also drives the HuggingFace backend: any `--model`
containing a `/` is auto-detected as a HuggingFace Hub ID. So a whole sweep is a
single cell. In a fresh Colab notebook:

```python
# 0. Get the code and dependencies
!git clone https://github.com/ThoHardy/LLM_Blink.git
!pip -q install "transformers>=4.44" accelerate
# !pip -q install bitsandbytes        # only for 7B 4-bit

# 1. Run a sweep (CSV is written next to the notebook)
!python LLM_Blink/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --n-seeds 10
# 7B in 4-bit:  !python LLM_Blink/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

`run_experiment.py --help` lists every option (`--lags`, `--loads`, `--regimes`,
`--n-pres`, `--n-seeds`, `--no-generation`, `--output`). Results land in
`ab_results_<model_slug>.csv`.

`--n-pres` is a confound-control dimension: by default it is `auto`
(n_pre auto-computed so T2 stays at the same absolute packet index, today's
behavior). Pass explicit ints (e.g. `--n-pres 2 4 6 8`) to vary the number of
pre-T1 filler packets and let T2 absolute position float — useful for
disentangling lag effects from position effects. The actually-used value is
recorded in the `n_pre` column of every output row.

To plot, read that CSV back in a notebook cell rather than passing `--plot`
(a `!python` subprocess can't render inline figures):

```python
import pandas as pd
from LLM_Blink import plot_ab          # repo is on sys.path after the clone
import matplotlib.pyplot as plt
df = pd.read_csv("ab_results_Qwen_Qwen2.5-3B-Instruct.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_ab(df, "t2_mean_logprob", ax=axes[0])
if "report_correct" in df.columns:
    plot_ab(df, "report_correct", ax=axes[1])
plt.tight_layout(); plt.show()
```

### Notebook path — `attentional_blink_colab.ipynb`

For interactive, cell-by-cell control, open `attentional_blink_colab.ipynb`
(inside the cloned repo) instead and follow the cells. The manual setup is below.

### 0. Install dependencies

```python
!pip -q install "transformers>=4.44" accelerate
# bitsandbytes only needed for 7B 4-bit:
# !pip -q install bitsandbytes
```

### 1. Import the toolbox

```python
import sys, os
sys.path.insert(0, os.getcwd())   # adjust if LLM_Blink/ is elsewhere
from LLM_Blink import load_model, run_sweep, plot_ab, build_trial, TrialConfig
print("LLM_Blink imported OK")
```

### 2. Load a model

```python
model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")   # fp16, fits T4
# Alternatives:
# load_model("Qwen/Qwen2.5-1.5B-Instruct")            # faster
# load_model("google/gemma-2-2b-it")
# load_model("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)  # needs bitsandbytes
```

Any HuggingFace model ID (containing `/`) selects the **HuggingFace backend** automatically.

---

## Quickstart — Local machine with Ollama

Run the same experiment on any model served by [Ollama](https://ollama.com), with no GPU required (CPU is fine for small models).

### 0. Install Ollama and pull a model

```bash
# macOS / Linux — see https://ollama.com/download for Windows
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b        # ~2 GB, good default
# other options: qwen2.5:3b, gemma2:2b, mistral:7b, phi3:mini …
```

### 1. Start the Ollama server (if not already running)

```bash
ollama serve
```

### 2. Install Python dependencies

```bash
pip install openai pandas matplotlib
# transformers / torch are NOT needed for the Ollama backend
```

### 3. Clone the repo and run

```bash
git clone <your-repo-url>
cd LLM_AB

# Pass your model name once — everything else uses sensible defaults.
python LLM_Blink/run_experiment.py --model gemma2:2b
python LLM_Blink/run_experiment.py --model mistral:7b --n-seeds 20
python LLM_Blink/run_experiment.py --model gemma3:12b --regimes cot direct --plot
```

Results are saved to `ab_results_<model>.csv` in the current directory.
Run `python LLM_Blink/run_experiment.py --help` for all options.

### Notes on the Ollama backend

- **Model selection:** pass the Ollama tag exactly as it appears in `ollama list`, e.g. `"llama3.2:3b"`, `"qwen2.5:3b"`, `"gemma2:2b"`.
- **Log-prob scoring** requires Ollama ≥ 0.3 (logprobs support). If your build is older, the `t2_mean_logprob` columns will be `NaN` and only the binary `report_correct` measure will be populated.
- **Custom server URL:** if Ollama runs on a different host/port, pass `ollama_base_url`:
  ```python
  model, tok = load_model("llama3.2:3b", ollama_base_url="http://192.168.1.10:11434/v1")
  ```
- The `tok` value returned is `None` for the Ollama backend; you never need to use it directly.

---

## Steps 3–6 (same for both backends)

### 3. Inspect one trial (sanity check)

```python
tr = build_trial(TrialConfig(lag=2, t1_load="semantic_3", regime="cot", seed=0))
print(tr.user_prefix)
print("\nT2 to detect:", tr.t2_phrase, "| T1 answer:", tr.t1_answer)
```

### 4. Run a pilot lag sweep

2 loads × 6 lags × 2 regimes × `n_seeds` trials. Start with `n_seeds=10`, scale up once it
looks right. `do_generation=True` also runs greedy decoding for the binary report measure
(slower).

```python
df = run_sweep(
    model, tok,
    lags=(0, 2, 4, 6, 8, 10),
    # Minimal Quickstart: baseline + hardest semantic level.
    # T1 load supports all 11 levels (semantic_0..4, math_0..4, none) for the full sweep.
    loads=("none", "semantic_4"),
    regimes=("cot", "direct"),   # contrast generation-dynamics vs encoding blink
    n_seeds=10,
    do_generation=True,
)
df.to_csv("ab_results.csv", index=False)
df.head()
```

### 5. Plot the AB curve

One panel per regime, with both measures stacked. CoT (generation dynamics) on one side,
direct (encoding) on the other.

```python
import matplotlib.pyplot as plt
regimes_present = sorted(df["regime"].unique())
has_correct = "report_correct" in df.columns
n_rows = 2 if has_correct else 1
fig, axes = plt.subplots(
    n_rows, len(regimes_present),
    figsize=(6 * len(regimes_present), 4 * n_rows),
    squeeze=False,
)
for col, regime in enumerate(regimes_present):
    plot_ab(df, "t2_mean_logprob", regime=regime, ax=axes[0, col])  # graded 'unconscious'
    if has_correct:
        plot_ab(df, "report_correct", regime=regime, ax=axes[1, col])  # binary 'conscious'
plt.tight_layout(); plt.show()
```

**What to look for:** `T1=semantic_4` should deepen and/or widen the dip at intermediate
lags if a blink-like effect exists; `T1=none` stays flat (positional baseline). The
CoT/direct contrast separates a generation-dynamics blink from an encoding one. All flat
→ informative null.

### 6. Sanity check — was T1 load actually performed?

```python
if "t1_correct" in df.columns:
    print(df.groupby("t1_load")["t1_correct"].mean(dropna=True))
```

If `t1_correct` is near 0 for `semantic_4`, the model isn't paying the load cost — interpret results with care.

---

## Package layout

```
LLM_Blink/
├── model.py       # load_model(), two backends (HF + Ollama), scorers
├── stimuli.py     # packet stream builder, T1/T2 generation, lag control
├── experiment.py  # TrialConfig, build_trial(), run_sweep()
└── analyze.py     # plot_ab(), aggregate helpers
```

## Next steps

- Run a difficulty sweep: `loads=("none","semantic_0","semantic_1","semantic_2","semantic_3","semantic_4")` to check that T1 accuracy decreases monotonically (sanity) and that the blink amplitude grows with level.
- Re-run with `regimes=("direct",)` and compare (encoding vs generation-dynamics blink).
- Add a second model family (e.g. `gemma2:2b` via Ollama or `gemma-2-2b-it` via HF).
- Titrate toward ~50% report rate via T2 length / mask / distractor similarity (see `../PROMPTS.md` P2).