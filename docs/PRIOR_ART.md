# Prior art and scope

## Quaternion and “4D” in machine learning

- **Quaternion-valued neural networks** (and broader hypercomplex / Clifford approaches) have a long line of research, especially in **signal processing**, **computer vision**, and some **sequence modeling** work. They are not new as a mathematical idea.
- **Mainstream LLM inference** today is dominated by **real-valued** transformers running in **llama.cpp**, **vLLM**, **Ollama**, vendor APIs, etc. None of those are “missing” quaternions by accident—industrial stacks optimize for **throughput and portability** on hardware that is built for FP16/BF16/INT8 real tensors.

## This repository

- **four_d_engine** is a **purpose-built** Rust core: Hamilton quaternions, 4D tensor helpers, GGUF inspection, lift **preview**—exposed over a **small C ABI** for the Go server.
- It is **not** a claim that the world’s first quaternion engine exists only here; it **is** a concrete, testable codebase that combines that math layer with an **Ollama-shaped** HTTP surface.
- **Autoregressive token generation** in this engine is **not** shipped yet. Calling the HTTP API “true 4D inference” in the same sense as Ollama would be **misleading** until decode is implemented and benchmarked.

## How to verify what you built

- `GET /api/engine` — JSON capability manifest from the linked native library (or stub message if `CGO_ENABLED=0`).
- `fd4_capabilities_json` / `fd4_gguf_inspect_json` — Rust FFI entry points (see `four_d_engine.h`).

## If the goal is “incredible” in a defensible way

Ship **measurements**: same model, same quant, same machine, **tokens/s** and **latency** vs a baseline, plus **correctness** checks on the 4D path. Adjectives without numbers are not engineering.
