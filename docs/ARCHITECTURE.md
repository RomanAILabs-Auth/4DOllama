# 4dollama architecture

This document is the engineering contract for what the codebase **does today**, what **“4D”** means here, and what would be required for claims about **speed** or **reasoning quality** to be defensible.

## Honest positioning

- **Reasoning depth** and **latency vs Ollama** are not something we can “invent” in application glue code. They come from **model weights**, **inference kernels** (CPU/GPU), **quantization**, **batching**, and **measured benchmarks** on your hardware.
- Marketing language like “first in the world” or “never been seen before” belongs **after** reproducible results (correctness, throughput, quality evals), not before.

## Is this “truly 4D architecture”?

**Yes, in a software / mathematical sense — no, in a physics sense.**

| Layer | What “4D” means here |
|-------|----------------------|
| **Rust `four_d_engine`** | Tensors and ops with a **fourth axis (w)** and **quaternion** paths; **Cl(4,0)-flavored** structure in converters and ops; GGUF **lift planning** and FFI for inspection. |
| **Go `internal/runner` + inference** | Wires **native** decode that uses **4D GEMM**, **spacetime-style attention**, **quaternion RoPE** where enabled — this is **architecturally** a 4D-augmented stack, not a 1D-only LM shell. |
| **RomanAI `.4dai`** | **JSON** (`romanai.4dai`) or **ROMANAI4+safetensors** shards; **Cl(4,0) carrier** blocks in specs and tools (e.g. RQ4D / Python `ai4d_format`). |
| **Optional Python `fourdollama`** | Generates **`.r4d` kernels** and calls **`r4d run`** — Roma4D **language** surface, separate from Go GGUF serving. |
| **Not claimed** | That the hardware executes in literal 4D spacetime, or that “4D” alone guarantees SOTA quality without evals. |

So: **4D-native *software architecture*** (extra geometric structure in the engine and file formats) is accurate; **4D physics in a box** is not.

## Layers

```
CLI (Cobra) ──► HTTP (chi) ──► Handler ──► runner.Service
                              │              │
                              │              ├── models.Registry (GGUF + .4dai + Ollama blob share)
                              │              ├── engine.Engine (GGUF inspect via CGO → Rust)
                              │              └── inference.Provider
                              │                     ├── Native — four_d_engine decode path
                              │                     └── OllamaForward — POST to OLLAMA_HOST
                              └── Streaming chunking (optional FOURD_STREAM_CHUNK_MS)
```

- **`4d-engine` (Rust):** GGUF inspection, tensor inventory, lift **preview**, quaternion / 4D tensor ops — exposed to Go via **`fd4_gguf_inspect_json`**. Full-speed **native** generation is implemented in the **Go+Rust** path the runner selects (`FOURD_INFERENCE`); **hybrid** forwarding is optional.
- **Go `internal/inference`:** Chooses how user-visible text is produced. **`ollama`** delegates to a real Ollama process for **upstream** completions (extra network hop). **`stub` / native** keeps decoding in-process with the **4D engine** hooks.

## CLI surface (Ollama parity)

The **`4dollama`** binary uses **github.com/spf13/cobra** with the same **verbs** as Ollama (`serve`, `run`, `pull`, `create`, `list`, `stop`, …). See **`docs/4DOllama.md`** and **`docs/4DOLLAMA_REFERENCE_FOR_LLMS.md`**.

## Environment

| Variable | Role |
|----------|------|
| `FOURD_INFERENCE` | `stub` / `fourd` / `native` vs **`ollama`** hybrid |
| `OLLAMA_HOST` | e.g. `http://127.0.0.1:11434` — hybrid / benchmarks |
| `FOURD_STREAM_CHUNK_MS` | Optional pacing for NDJSON streaming; default `0` |
| `FOURD_MODELS` | `~/.4dollama/models` — **GGUF** and **`.4dai`** |

## Roadmap (technical, not hype)

1. **Tighter native path:** Continue fusing **llama-compatible** decode with **4D ops** where numerically validated.
2. **4D-specific compute:** Where quaternion / w-axis ops materially change activations, lock in with **golden tests** vs reference.
3. **Performance:** Profile end-to-end; compare to Ollama/llama.cpp with **identical model + quant + GPU**; publish numbers, not adjectives.

## Professional bar for “groundbreaking”

- Public **repro steps**, **hardware spec**, **commit hash**, and **metrics** (tokens/s, TTFT, memory).
- Clarity on **what is experimental** vs **production-safe**.

This repository is structured so those milestones can be added **without** rewriting the HTTP surface.
