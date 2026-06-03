# Attentional Blink for LLMs

Tests whether an **Attentional-Blink (AB)-like phenomenon** exists in LLMs, using a deliberately operationalized notion of "conscious access":

- **Conscious perception** of T2 = the item appears in the model's tokenized output (it is reported).
- **Unconscious strength** = joint log-probability of the correct T2 under the forced template, regardless of what was generated.

See [`../LITERATURE.md`](../LITERATURE.md) for the design rationale and feasibility caveats.

---

## Quickstart — Google Colab (free T4 GPU)

Open `attentional_blink_colab.ipynb` in Colab: **Runtime → Change runtime type → T4 GPU**.

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
tr = build_trial(TrialConfig(lag=2, t1_load="hard", regime="cot", seed=0))
print(tr.user_prefix)
print("\nT2 to detect:", tr.t2_phrase, "| T1 answer:", tr.t1_answer)
```

### 4. Run a pilot lag sweep

3 loads × 6 lags × `n_seeds` trials. Start with `n_seeds=10`, scale up once it looks right.
`do_generation=True` also runs greedy decoding for the binary report measure (slower).

```python
df = run_sweep(
    model, tok,
    lags=(0, 1, 2, 3, 5, 8),
    loads=("none", "easy", "hard"),
    regimes=("cot",),       # try ("direct",) for the encoding regime
    n_seeds=10,
    do_generation=True,
)
df.to_csv("ab_results.csv", index=False)
df.head()
```

### 5. Plot the AB curve

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_ab(df, "t2_mean_logprob", ax=axes[0])    # graded 'unconscious'
if "report_correct" in df.columns:
    plot_ab(df, "report_correct", ax=axes[1]) # binary 'conscious'
plt.tight_layout(); plt.show()
```

**What to look for:** `T1=hard` dips at lag 2–3 and recovers by lag 6–8; `T1=none` stays flat (positional baseline). All flat → informative null.

### 6. Sanity check — was T1 load actually performed?

```python
if "t1_correct" in df.columns:
    print(df.groupby("t1_load")["t1_correct"].mean(dropna=True))
```

If `t1_correct` is near 0 for `hard`, the model isn't paying the load cost — interpret results with care.

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

- Re-run with `regimes=("direct",)` and compare (encoding vs generation-dynamics blink).
- Add a second model family (e.g. `gemma2:2b` via Ollama or `gemma-2-2b-it` via HF).
- Titrate toward ~50% report rate via T2 length / mask / distractor similarity (see `../PROMPTS.md` P2).
