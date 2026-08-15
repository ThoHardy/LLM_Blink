# Machine — local benchmarking host

Hardware and runtime configuration for the local Ollama sweeps in this repo.
Recorded 2026-07-17. Referenced by the smoke-test / full-run timings so results
are reproducible and per-model runtimes are interpretable.

## Hardware

| Component | Spec |
|-----------|------|
| Chip | Apple M4 Max |
| CPU cores | 16 (12 performance + 4 efficiency) |
| GPU cores | 40 (Metal 4) |
| Unified memory | 128 GB |
| Storage | 7.3 TB total, ~5.7 TB free |

Unified memory is the key enabler: 128 GB holds even the 27B (Q4, ~17 GB) models
with room to spare, and lets Ollama keep up to 2 models resident at once
(`OLLAMA_MAX_LOADED_MODELS=2`).

## OS & software

| Item | Version |
|------|---------|
| macOS | 26.5.1 (build 25F80) |
| Ollama | 0.30.10 |
| Python (`.venv`) | 3.9.6 |
| Backend deps | `openai`, `pandas`, `matplotlib` (no torch/transformers needed for Ollama) |

## Ollama runtime configuration

The GUI Ollama.app (Electron) does **not** inherit `launchctl setenv` variables,
so the server is run manually from a terminal with concurrency enabled:

```bash
export OLLAMA_NUM_PARALLEL=8        # batch up to 8 concurrent requests per model
export OLLAMA_MAX_LOADED_MODELS=2   # keep 2 models resident
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

Verify the live server picked these up:

```bash
grep "server config" results/ollama_serve.log   # shows OLLAMA_NUM_PARALLEL:8 ...
```

To revert to the normal desktop app, quit the terminal `ollama serve`
(`pkill -f "ollama serve"`) and relaunch Ollama.app.

## Why parallelism helps here

`run_sweep` issues trials sequentially (one HTTP request at a time), which leaves
the 40-core GPU underused on small/mid models. Trials are independent, so
`results/parallel_sweep.py` dispatches them across a thread pool (the Ollama
backend is pure HTTP I/O via a thread-safe shared client). With
`OLLAMA_NUM_PARALLEL=8` the server batches the concurrent decodes on the GPU,
amortising the memory-bandwidth-bound weight reads across sequences.

At `temperature=0` the parallel driver is numerically equivalent to the
sequential one: greedy generation is bit-identical (so `report_correct`,
`t1_correct`, all quality flags and stream geometry match exactly), and the
teacher-forced `t2_*_logprob` differs only by floating-point batching noise
(max |Δ| ≈ 0.04 on total logprob, correlation 0.9999999, grouped means identical
to 4 decimals). Safe to use for the graded measure.

## Measured throughput (smoke test: 240 trials = 6 lags × 2 loads × 2 regimes × 10 seeds)

| Model | Params | Sequential | Parallel (8-way) | Notes |
|-------|-------:|-----------:|-----------------:|-------|
| gemma3:270m | 0.27B | 118 s (0.49 s/trial) | — | degenerate: never follows format |
| gemma3:1b | 1B | 237 s (0.99 s/trial) | — | |
| gemma2:2b | 2B | 393 s (1.64 s/trial) | 310 s (1.29 s/trial) | validation model |
| gemma3:4b | 4B | — | *(measuring)* | |
| gemma2:9b | 9B | — | *(measuring)* | |
| gemma3:12b | 12B | — | *(measuring)* | |
| gemma2:27b | 27B | — | *(measuring)* | |
| gemma3:27b | 27B | — | *(measuring)* | |

Parallel speedup grows with model size (bigger models are more
bandwidth-bound during decode, so batching pays off more). This table is
updated as the fleet completes.
