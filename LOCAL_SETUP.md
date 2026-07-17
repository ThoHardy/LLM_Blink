# Running LLM_Blink locally (Windows, RTX 1000 Ada 6 GB)

Machine: i7-13850HX (20 cores) · 32 GB RAM · NVIDIA RTX 1000 Ada Laptop GPU (6 GB VRAM,
CUDA compute 8.9) · Windows 11. Plenty for a *light* Qwen via the HuggingFace backend.

## Model sizing on 6 GB VRAM (fp16)
| Model                     | VRAM (fp16) | Fits GPU? | Notes                                   |
|---------------------------|-------------|-----------|-----------------------------------------|
| Qwen2.5-0.5B-Instruct     | ~1.5 GB     | yes       | Fastest; good for wiring/smoke tests.   |
| Qwen2.5-1.5B-Instruct     | ~3.5 GB     | yes       | **Recommended** light model for GPU.    |
| Qwen2.5-3B-Instruct fp16  | ~6.5 GB     | no        | Weights alone exceed 6 GB + KV cache.   |
| Qwen2.5-3B-Instruct 4-bit | ~2.5 GB     | yes       | Needs bitsandbytes (`--load-in-4bit`).  |
| CPU (any size)            | uses 32 GB  | n/a       | Works, slower; 3B fine on CPU.          |

## Backend choice
Use the **HuggingFace/transformers** backend (model name contains a `/`), not Ollama.
The study's graded "unconscious strength" read-out is an exact teacher-forced joint
log-prob — HF gives that cleanly; the Ollama backend's logprobs are approximate and can
return NaN. Ollama is fine only for quick binary-report checks.

## One-time setup (PowerShell)
```powershell
cd C:\Users\thoma\Documents\These\LLM_AB
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA 12.1 build of torch for the RTX 1000 Ada:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.44" accelerate pandas matplotlib
# optional, only for 3B 4-bit:
# pip install bitsandbytes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect `True NVIDIA RTX 1000 Ada Generation Laptop GPU`.

## Preview one prompt (no GPU, no download)
```powershell
python -c "import sys;sys.path.insert(0,'.');from LLM_Blink import build_trial,TrialConfig;print(build_trial(TrialConfig(lag=2,t1_load='semantic_4',regime='cot',seed=0)).user)"
```

## Run the default sweep (first run downloads ~3 GB to the HF cache)
```powershell
python LLM_Blink\run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --n-seeds 10
# tiny/fast smoke run:
python LLM_Blink\run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --n-seeds 1 --lags 0 4
# 3B in 4-bit:
python LLM_Blink\run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit --n-seeds 10
```
Default sweep = lags(0 2 4 6 8 10) x loads(none, semantic_4) x regimes(cot, direct) x 10 seeds
= 240 trials. On this GPU, 1.5B ~ roughly 10-20 min (rough estimate); 0.5B faster; 3B/4-bit slower.
Output: `ab_results_Qwen_Qwen2.5-1.5B-Instruct.csv` next to where you run it.

## Sanity checks before interpreting (see README "Sanity checks & controls")
- `thinking_is_placeholder` copy-rate for cot rows (Qwen often skips real CoT).
- `t1_correct` per load must fall with difficulty (was the load actually paid?).
- `t2_slot_missing` — gate graded-score rows on it.
