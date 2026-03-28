# 4DOllama

**4DOllama** is an **Ollama-compatible** HTTP API and command-line tool that serves GGUF models through a **native 4D numerical stack** (quaternion RoPE, spacetime attention, lattice coupling, and autoregressive decoding), while exposing the same routes and verbs developers already use with [Ollama](https://github.com/ollama/ollama). It is designed to run **beside** Ollama on a dedicated port by default, or on **Ollama’s port** when you explicitly configure it.

This directory (`4DOllama/`) also ships an **optional Python package** (`fourdollama`) for a lightweight HTTP bridge to Roma4D tooling; the **primary product** is the **Go** binary **`4dollama`** built from the monorepo root (`cmd/4dollama`).

---

## What 4DOllama does

1. **Registry and pulls** — Resolves model names, lists tags, and pulls GGUF artifacts from the Ollama registry (with optional reuse of blobs under `~/.ollama/models` when enabled).
2. **HTTP API** — Implements the familiar Ollama surface: `/api/generate`, `/api/chat`, `/api/tags`, `/api/pull`, `/api/show`, embeddings, OpenAI-shaped `/v1/*` routes, health checks, and metrics suitable for production deployments.
3. **Native inference path (default)** — With `FOURD_INFERENCE` unset or set to `stub` / `fourd` / `native`, decoding runs **in-process**: GGUF is located under `FOURD_MODELS`, weights are sampled and lifted, and each autoregressive step flows through **RoPE → SpacetimeAttention4D → logits**, with an optional **4D lattice** coupling (`internal/inference/coupling_lattice.go`) that feeds back as **logit bias** (tunable via `FOURD_LATTICE_KAPPA`).
4. **Optional hybrid mode** — Set `FOURD_INFERENCE=ollama` and `OLLAMA_HOST` to forward completions to a stock Ollama server while keeping 4DOllama’s API and CLI; useful for A/B or migration.

---

## How it works (architecture)

| Layer | Responsibility |
|--------|----------------|
| **CLI** (`internal/cli`) | Parses Ollama-shaped commands (`serve`, `run`, `pull`, `list`, `ps`, `show`, `cp`, `rm`, `version`, …). Interactive **`4dollama run <model>`** uses a plain **`>>> `** line REPL (no full-screen TUI). |
| **HTTP server** (`internal/httpserver`) | Routes, middleware (request IDs, recovery), streaming writers. Long-lived responses use **chunked NDJSON** with periodic flush so clients see tokens as they are produced. |
| **Runner** (`internal/runner`) | Orchestrates model resolution, GGUF inspect, inference context construction, and **streaming callbacks** for chat and generate. |
| **Inference** (`internal/inference`) | Pluggable providers: native **stub** path (default) vs **ollama-forward** for hybrid. |
| **Engine** (`internal/engine`) | **CGO** bridge to the Rust **`four_d_engine`** when built with `cgo` and the release library; otherwise a safe **stub** engine for CI and minimal builds. |
| **4D substrate** (`internal/fourd`) | Clifford algebra helpers, lattice dynamics, Q-tensor style coupling hooks, and the **`4dollama fourd`** subcommand for demos (`ga-demo`, `lattice`, …). |

**Request flow (streaming chat):** the client sends `POST /api/chat` with `"stream": true`. The handler sets `Content-Type: application/x-ndjson`, streams one JSON object per line (`message.content` deltas, then `done: true`), and **flushes** after each write so interactive CLIs and curl see incremental output without waiting for the full completion.

---

## Streaming (first-class)

4DOllama treats **streaming** as the normal path for interactive use:

- **HTTP** — `stream: true` on `/api/chat` and `/api/generate` returns **newline-delimited JSON** (NDJSON). Each delta is written and flushed as soon as it is available from the inference backend (native stub or forwarded Ollama chunks in hybrid mode).
- **CLI** — `4dollama run <model>` in a terminal consumes that stream over HTTP and prints assistant text incrementally to stdout.
- **Tuning** — `FOURD_STREAM_CHUNK_MS` adds an optional delay between chunks (milliseconds) for smoother display in some terminals; default is **0** (no artificial delay).

For strict Ollama parity, non-streaming requests are also supported: omit `stream` or set `stream: false` to receive a single JSON body when the full response is ready.

---

## Languages and codebase

| Language | Role in 4DOllama |
|----------|------------------|
| **Go** (1.22+) | Main **CLI**, **HTTP server**, configuration, model registry, runner, optional **fourd** lattice packages under `internal/fourd/`, and tests. Go was chosen for fast static binaries, excellent concurrency for many concurrent streams, and straightforward deployment. |
| **Rust** | The **`four_d_engine`** crate under `4d-engine/`: quaternion RoPE, 4D attention, GEMM helpers, GGUF sampling/lift, and FFI exported to Go via **CGO**. Rust provides memory-safe numerical kernels and a single artifact (`libfour_d_engine`) linked at build time when CGO is enabled. |
| **Python** (3.10+, optional) | The **`fourdollama`** package in this folder: Typer-based CLI and optional **FastAPI**/Uvicorn bridge (default **127.0.0.1:13377** via `FOURDOLLAMA_PORT`) for workflows that integrate with **Roma4D** (`r4d`). This is **not** required for the Go `4dollama` server. |

**Roma4D** (the `.r4d` language and `r4d` compiler in `roma4d/`) lives in the **same monorepo** but is a **separate product**. 4DOllama does not compile Roma4D source as part of inference; the Python bridge can optionally invoke `r4d` for kernel or demo workflows. For GGUF chat and the Ollama API, use the **Go** binary.

---

## Quick start (Go `4dollama`, monorepo root)

From the repository root (next to `go.mod`):

```bash
go build -o 4dollama ./cmd/4dollama
./4dollama serve
# Default: http://127.0.0.1:13377 (set FOURD_HOST/FOURD_PORT as needed)
```

In another terminal:

```bash
./4dollama pull qwen2.5
./4dollama run qwen2.5
```

**Windows (one-click):** from repo root, `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1` builds the engine when possible, installs `4dollama.exe` under `%USERPROFILE%\.4dollama\bin`, sets common environment variables, and can start `serve` in the background. See **[../docs/4DOllama.md](../docs/4DOllama.md)** for full tables and Docker.

**4D demos (Go):**

```bash
./4dollama fourd ga-demo
./4dollama fourd lattice -steps 80 -kappa 0.002 -inject-every 8
```

**Lattice / Cl(4,0) roadmap and math notes:** [docs/NATIVE_4D_ENGINE.md](docs/NATIVE_4D_ENGINE.md).

---

## Ports and running next to Ollama

| | Ollama | 4DOllama (default) |
|--|--------|---------------------|
| Typical URL | `http://127.0.0.1:11434` | `http://127.0.0.1:13377` |
| CLI | `ollama` | `4dollama` |

- **`FOURD_PORT`** defaults to **13377** so both servers can run without conflict.
- To **bind Ollama’s port**, stop Ollama and e.g. `4dollama serve -h 127.0.0.1 -p 11434` (or set `FOURD_PORT=11434`).
- **`FOURD_DROPIN=1`** sets the listen address to **127.0.0.1** when `FOURD_HOST` is not explicitly set; it does **not** change the port away from **13377** unless you set **`FOURD_PORT`**.

The optional Python CLI uses **`FOURDOLLAMA_PORT`** (default **13377**). Keep it aligned with `FOURD_PORT` when you point **`fourdollama --remote`** at the Go API.

---

## Configuration (essentials)

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOURD_HOST` | `0.0.0.0` | Bind address |
| `FOURD_PORT` | `13377` | HTTP port |
| `FOURD_MODELS` | `~/.4dollama/models` | GGUF tree |
| `FOURD_LOG_LEVEL` / `LOG_LEVEL` | `warn` | Quiet stderr; use `debug` or `4dollama serve -verbose` for diagnostics |
| `FOURD_INFERENCE` | `stub` | Native path; `ollama` + `OLLAMA_HOST` for hybrid |
| `FOURD_LATTICE_KAPPA` | (see code) | Strength of lattice → logit bias coupling |
| `FOURD_STREAM_CHUNK_MS` | `0` | Optional delay between streamed NDJSON chunks |

Extended reference: **[../docs/4DOllama.md](../docs/4DOllama.md)**.

---

## Optional Python package (`fourdollama`)

For the Roma4D-aware HTTP bridge and Typer CLI:

```bash
cd 4DOllama
pip install -e .
python -m fourdollama serve    # FastAPI on 13377 by default
python -m fourdollama run MODEL --remote   # stream via Go /api/chat
```

Environment: `FOURDOLLAMA_HOST`, `FOURDOLLAMA_PORT`, `FOURDOLLAMA_R4D`, `R4D_PKG_ROOT`, etc. **GGUF + native decode** for production chat should still use **`4dollama`** from `cmd/4dollama`.

---

## Troubleshooting

- **Stale UI** (pink header, duplicate “Message…” chrome): caused by an **old `4dollama.exe`** on `PATH`. Run `where.exe 4dollama` (Windows) or `which -a 4dollama`, then rebuild from this repo and prefer `%USERPROFILE%\.4dollama\bin\4dollama.exe` after `scripts/install.ps1`.
- **Connection refused**: ensure `4dollama serve` is running on the same host/port your CLI or `fourdollama --remote` targets (**13377** by default).

---

## Documentation index

| Document | Content |
|----------|---------|
| [../docs/4DOllama.md](../docs/4DOllama.md) | Install, API table, curl examples, Docker, logging |
| [docs/NATIVE_4D_ENGINE.md](docs/NATIVE_4D_ENGINE.md) | Native 4D engine and lattice roadmap |
| [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) | Monorepo architecture |

---

## License and credits

See the monorepo **LICENSE** / **NOTICE** files and **RomanAILabs** attribution in source headers. **4DOllama** and **Roma4D** are related projects in this tree; trademarks apply to names and branding as described in repository license text.

## Contact

**RomanAILabs** — *Daniel Harding*  
[romanailabs@gmail.com](mailto:romanailabs@gmail.com) · [daniel@romanailabs.com](mailto:daniel@romanailabs.com)
