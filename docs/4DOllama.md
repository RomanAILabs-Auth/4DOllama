# 4DOllama (`4dollama`)

This document preserves the **4DOllama** product guide for the **4DEngine** monorepo. The repository’s main [README](../README.md) leads with **Roma4D**; this file is the home for Ollama-style CLI, HTTP API, and `four_d_engine` build notes.

## Ollama-identical interactive chat (Phase 11)

`4dollama run <model>` matches **`ollama run`**-style usage: a plain **`>>> `** line REPL on normal stdout (scrollback and text selection work like any terminal app—no full-screen TUI). Assistant text **streams over `/api/chat`**: the server flushes NDJSON **as each token is produced** (native stub) or **as upstream Ollama chunks arrive** (hybrid), not after buffering the full reply. **`/help`** prints one short line on stderr; **`/clear`**, **`/bye`** (also **`/exit`** / **`/quit`**). Consecutive duplicate user messages are dropped server-side. If you still see a pink header or duplicate “Message… Ollama-style” hints, that is a **stale `4dollama.exe`** on PATH—rebuild this repo and put that binary first.

**Logging:** Quaternion RoPE, spacetime attention, 4D GEMM, and related engine traces are **`debug`** logs only. Use **`4dollama serve -verbose`**, **`FOURD_LOG_LEVEL=debug`**, or **`LOG_LEVEL=debug`** to print them. When **`run`** auto-starts **`serve`** for interactive chat (terminal UI), the child uses **`FOURD_LOG_LEVEL=error`** so stderr stays quiet.

**System context:** Every **`/api/chat`** request gets a permanent **system** preamble (merged with any client `system` message) describing quaternion RoPE, spacetime attention, 4D GEMM, and GGUF lift so the model can answer product questions accurately.

---

**One-click install** (from repo root) builds the Rust `four_d_engine`, compiles the Go CLI (CGO when possible, stub fallback), sets **`FOURD_GPU=cpu`** when no GPU path is detected, pulls **qwen2.5** (best effort), starts **`4dollama serve`** in the background, and prints:

> 4DOllama is ready! Works on CPU or GPU. Just type: `4dollama run qwen2.5`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
# Optional: winget Go — .\scripts\install.ps1 -InstallGo
```

```bash
chmod +x scripts/install.sh && ./scripts/install.sh
```

After install, open a **new** terminal if your shell PATH was updated. **`4dollama doctor`** confirms listen URL, model dir, and **“CPU mode active”** on CPU-only hosts. **`4dollama run qwen2.5`** (no extra args) always uses the **line-based** prompt (`>>> `, **Enter** to send), including under **Windows / Cursor / VS Code**. If the API is down, `run` auto-starts `serve` from the same binary. **Rebuild** after pulling UI fixes (`go build -o 4dollama.exe ./cmd/4dollama` or `scripts/install.ps1`) — an old `4dollama.exe` will still show the removed bubble UI.

Production-oriented **Ollama-compatible** HTTP API and CLI with a pluggable **4D engine** (`four_d_engine`, Rust) exposed over **CGO**. The server is suitable for containers (non-root user, health checks, structured logs, request limits, graceful shutdown). Docker image sets **`FOURD_GPU=cpu`** by default (typical CPU-only container).

**Scope today:** **Default inference is native four_d_engine** (`FOURD_INFERENCE` unset or `stub|fourd|native`): pulled **GGUF** is resolved under `FOURD_MODELS`, lifted/sampled, then **autoregressive decode** runs through quaternion RoPE → SpacetimeAttention4D → projected logits in-process (no llama.cpp). **Hybrid** is **opt-in only**: set **`FOURD_INFERENCE=ollama`** and **`OLLAMA_HOST`** to forward completions to stock Ollama while keeping 4dollama’s API on **13373**. See `docs/ARCHITECTURE.md`.

## Why a 4D engine?

- **Quaternion RoPE path:** spatial rotations via `Quaternion::rotate_vec3` instead of complex phase alone (`4d-engine` notes in source).
- **4D GEMM:** `gemm4d_w_contract` contracts over the **w** axis (`tensor4d.rs`).
- **GGUF → 4D lift planning:** `converter` estimates memory / tensor inventory before native `.4dgguf` export lands.
- **GPU scaffold:** `gpu::cuda_backend_label()` (CUDA kernels stubbed for hosts without NVCC).

## Layout

```
cmd/4dollama/          # CLI entrypoint
internal/
  httpserver/          # chi router, middleware, handlers
  runner/              # generate/chat orchestration
  inference/           # stub vs ollama-forward providers
  models/              # on-disk GGUF registry
  engine/              # cgo ↔ four_d_engine (or stub without cgo)
  config/, ollama/, cli/, version/
4d-engine/             # Rust: tensor4d, ops, converter, ffi, gpu
roma4d/                # Roma4D language (see repository README)
```

## API (subset)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/healthz`, `/livez` | Liveness / JSON health |
| GET | `/metrics` | Prometheus text (basic counters) |
| GET | `/api/version` | Server version |
| GET | `/api/engine` | Server + **four_d_engine** capability JSON |
| GET | `/api/tags` | Lists `*.gguf` under `FOURD_MODELS` |
| GET | `/api/ps` | Stub (empty) |
| POST | `/api/pull` | Pull from Ollama registry (NDJSON stream) |
| POST | `/api/generate`, `/api/chat`, `/api/embeddings` | Ollama-shaped JSON |
| POST | `/v1/chat/completions`, `/v1/completions` | OpenAI-shaped JSON |

### Example: curl

Server listens on **13373** (not 11434). Use model **qwen2.5** when sharing Ollama blobs.

```bash
# Terminal A
FOURD_MODELS=./models FOURD_LOG_LEVEL=info ./4dollama serve

# Terminal B
curl -s http://127.0.0.1:13373/api/tags | jq .
curl -s http://localhost:13373/api/generate -d '{"model":"qwen2.5","prompt":"Hello 4D world!"}' | jq .
```

## Configuration (12-factor)

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOURD_HOST` | `0.0.0.0` | Bind address |
| `FOURD_PORT` | `13373` | Port (**not** 11434 — runs beside Ollama) |
| `FOURD_MODELS` | `~/.4dollama/models` | GGUF search path |
| `FOURD_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` (if unset, **`LOG_LEVEL`** is used) |
| `LOG_LEVEL` | _(see above)_ | Optional alias when `FOURD_LOG_LEVEL` is empty |
| `FOURD_LOG_JSON` | `false` | JSON logs to stderr |
| `OLLAMA_HOST` | _(empty)_ | Baseline URL for `benchmark-4d` (Windows `install.ps1` sets `http://127.0.0.1:11434`) |
| `FOURD_BENCH_MODEL` | _(auto)_ | Model name for benchmarks |
| `FOURD_SHARE_OLLAMA` | `true` | When true, **list/resolve** scan Ollama’s `manifests/` + use **`blobs/`** paths |
| `OLLAMA_MODELS` | `~/.ollama/models` | Ollama data root (manifests + blobs) |
| `FOURD_DEFAULT_MODEL` | `qwen2.5` | Install script / docs hint only |
| `FOURD_GPU` | _(auto)_ | Install scripts set **`cpu`** when no discrete GPU / CUDA path is found |
| `FOURD_INFERENCE` | **`stub`** (default) | Native 4D decode on GGUF. Set **`ollama`** + **`OLLAMA_HOST`** only for hybrid forwarding. Aliases: `fourd`, `native`, `engine` → stub. |
| `FOURD_STREAM_CHUNK_MS` | `0` | Optional artificial delay between NDJSON chunks when `stream: true` |

CLI flag `-fourd-mode` maps the spec’s `--4d-mode` (Go’s `flag` cannot register `-4d-mode`).

## Building 4DOllama

### Rust

```bash
cd 4d-engine && cargo test && cargo build --release
```

### Go — with native engine (CGO)

```bash
export CGO_ENABLED=1
export LD_LIBRARY_PATH="$(pwd)/4d-engine/target/release:${LD_LIBRARY_PATH:-}"
go build -o 4dollama ./cmd/4dollama
```

### Go — stub engine

```bash
CGO_ENABLED=0 go build -o 4dollama-stub ./cmd/4dollama
```

On Windows, use **MSVC** or **GNU** Rust targets; link `four_d_engine` accordingly or build the stub.

### Windows / Linux / macOS: `scripts/install.ps1` & `scripts/install.sh`

- **`cargo build --release`** for `4d-engine` when `cargo` is available; otherwise Go builds with **`CGO_ENABLED=0`** (stub engine still runs the full API surface).
- Copies **`4dollama`** to **`%USERPROFILE%\.4dollama\bin`** (Windows) or **`~/.local/bin`** (Unix) and updates PATH where applicable.

### Makefile

```bash
make rust
make go-stub
make docker
```

### Docker / Compose

```bash
docker build -t fourdollama:latest .
docker run --rm -p 13373:13373 -v ~/4d-models:/models fourdollama:latest
docker compose up --build
```

### Scripts

- `scripts/install.ps1` / `install.sh` — one-click build, CPU fallback, pull **qwen2.5**, background **serve**.
- `scripts/uninstall.ps1` / `uninstall.sh` — remove binaries and optional env/PATH entries.
- `scripts/install-linux.sh` — Docker image build when Docker is present.

### CI

GitHub Actions (`.github/workflows/ci.yml`): `cargo test`, `go test` (stub), `go build` with CGO against the freshly built `.so`.

## License (4DOllama / 4DEngine tooling)

MIT License — see repository root and file headers for copyright lines.

---

*This file was split from the former monolithic README so the root README can foreground Roma4D.*
