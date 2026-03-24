# 4dollama architecture

This document is the engineering contract for what the codebase **does today**, what **“4D”** means here, and what would be required for claims about **speed** or **reasoning quality** to be defensible.

## Honest positioning

- **Reasoning depth** and **latency vs Ollama** are not something we can “invent” in application glue code. They come from **model weights**, **inference kernels** (CPU/GPU), **quantization**, **batching**, and **measured benchmarks** on your hardware.
- Marketing language like “first in the world” or “never been seen before” belongs **after** reproducible results (correctness, throughput, quality evals), not before.

## Layers

```
CLI / TUI ──► HTTP (chi) ──► Handler ──► runner.Service
                              │              │
                              │              ├── models.Registry (GGUF paths, Ollama blob share)
                              │              ├── engine.Engine (GGUF inspect via CGO → Rust)
                              │              └── inference.Provider
                              │                     ├── Stub — deterministic demo text
                              │                     └── OllamaForward — POST to OLLAMA_HOST/api/generate
                              └── Streaming chunking (optional FOURD_STREAM_CHUNK_MS)
```

- **`4d-engine` (Rust):** GGUF inspection, tensor inventory, lift **preview**, quaternion / 4D tensor ops — exposed to Go via **`fd4_gguf_inspect_json`** (see `four_d_engine.h`). There is **no** C ABI yet for autoregressive token generation.
- **Go `internal/inference`:** Chooses how user-visible text is produced. **`stub`** keeps the server self-contained for API testing; **`ollama`** delegates to a real Ollama process for **actual LLM completions** (extra network hop → not faster than talking to Ollama directly).

## Environment

| Variable | Role |
|----------|------|
| `FOURD_INFERENCE` | Unset: **`ollama`** if `OLLAMA_HOST` is set, else **`stub`**. Explicit `stub` / `ollama` overrides. |
| `OLLAMA_HOST` | e.g. `http://127.0.0.1:11434` — required for `ollama` inference mode (auto-selected when set) |
| `FOURD_STREAM_CHUNK_MS` | Optional pacing for NDJSON streaming; default `0` |

## Roadmap (technical, not hype)

1. **Native decode path:** Implement token loop in Rust (or bind **llama.cpp** / similar) behind a second FFI surface; wire `runner` to select `native` vs `stub` vs `ollama`.
2. **4D-specific compute:** Where quaternion / w-axis ops materially change activations, integrate them into that decode path with **numerical tests** against a reference.
3. **Performance:** Profile end-to-end; compare to Ollama/llama.cpp with **identical model + quant + GPU**; publish numbers, not adjectives.

## Professional bar for “groundbreaking”

- Public **repro steps**, **hardware spec**, **commit hash**, and **metrics** (tokens/s, TTFT, memory).
- Clarity on **what is experimental** vs **production-safe**.

This repository is structured so those milestones can be added **without** rewriting the HTTP surface.
