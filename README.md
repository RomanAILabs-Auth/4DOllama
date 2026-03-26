# 4DOllama (`4dollama`)

## Ollama-identical interactive chat (Phase 11)

`4dollama run <model>` matches **`ollama run`**-style usage: **streaming** assistant text, **`/help`**, **`/clear`**, **`/bye`** (also **`/exit`** / **`/quit`**), **Ctrl+Enter** for a newline in the full-screen TUI, **PgUp** / **PgDown** to scroll history, and the status line: *Message... Enter = send (Ollama-style) · Ctrl+Enter = newline · /help /clear /bye*. Consecutive duplicate user messages are dropped server-side.

**Logging:** Quaternion RoPE, spacetime attention, 4D GEMM, and related engine traces are **`debug`** logs only. Use **`4dollama serve -verbose`**, **`FOURD_LOG_LEVEL=debug`**, or **`LOG_LEVEL=debug`** to print them. When **`run`** auto-starts **`serve`** for interactive chat (terminal UI), the child uses **`FOURD_LOG_LEVEL=error`** so stderr stays quiet.

**System context:** Every **`/api/chat`** request gets a permanent **system** preamble (merged with any client `system` message) describing quaternion RoPE, spacetime attention, 4D GEMM, and GGUF lift so the model can answer product questions accurately.

---

**One-click install** (from repo root) builds the Rust `four_d_engine`, compiles the Go CLI (CGO when possible, stub fallback), sets **`FOURD_GPU=cpu`** when no GPU path is detected, pulls **qwen2.5** (best effort), starts **`4dollama serve`** in the background, and prints:

> 🎉 4DOllama is ready! Works on CPU or GPU. Just type: `4dollama run qwen2.5`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
# Optional: winget Go — .\scripts\install.ps1 -InstallGo
```

```bash
chmod +x scripts/install.sh && ./scripts/install.sh
```

After install, open a **new** terminal if your shell PATH was updated. **`4dollama doctor`** confirms listen URL, model dir, and **“CPU mode active”** on CPU-only hosts. **`4dollama run qwen2.5`** (no extra args) opens **Ollama-style** chat: full-screen TUI when stdin is a real console TTY, or a **line-based** prompt (`>>> `, **Enter** to send) when stdout is a terminal but stdin is not — common on **Windows / Cursor / VS Code** integrated terminals. Use **`FOURD_LINE_CHAT=1`** to force line mode. **Enter** sends in the TUI (**Ctrl+Enter** inserts a newline). If the API is down, `run` auto-starts `serve` from the same binary. Rebuild and copy the new `4dollama` onto your PATH after pulling changes (`go build -o 4dollama.exe ./cmd/4dollama` or re-run `scripts/install.ps1`).

Production-oriented **Ollama-compatible** HTTP API and CLI with a pluggable **4D engine** (`four_d_engine`, Rust) exposed over **CGO**. The server is suitable for containers (non-root user, health checks, structured logs, request limits, graceful shutdown). Docker image sets **`FOURD_GPU=cpu`** by default (typical CPU-only container).

**Scope today:** real **GGUF metadata scan + lift preview** via Rust; **HTTP/CLI parity** for core routes; **inference is pluggable**. If **`OLLAMA_HOST`** is set (install scripts set `http://127.0.0.1:11434`), completions default to **`FOURD_INFERENCE=ollama`** — real answers via your **Ollama** process. With no `OLLAMA_HOST`, default is **stub** (deterministic demo tokens for API testing, not model-quality text). Set **`FOURD_INFERENCE=stub`** explicitly to keep demo mode even when Ollama is configured. Full native autoregressive decode in the Rust 4D stack is **roadmapped**; see `docs/ARCHITECTURE.md`.

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
```

### Roma4D (`roma4d/`)

**[Roma4D](https://github.com/RomanAILabs-Auth/Roma4D)** (4D spacetime language, `r4d` / `roma4d` CLI) is developed alongside this repo. Sources under **`roma4d/`** are a separate Go module; releases and primary git history live at **https://github.com/RomanAILabs-Auth/Roma4D**. After pulling compiler changes, run **`go install ./cmd/r4d ./cmd/roma4d`** from `roma4d/` (see that folder’s README for Windows / MinGW notes).

## API (subset)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/healthz`, `/livez` | Liveness / JSON health |
| GET | `/metrics` | Prometheus text (basic counters) |
| GET | `/api/version` | Server version |
| GET | `/api/engine` | Server + **four_d_engine** capability JSON (quaternion/4D FFI manifest) |
| GET | `/api/tags` | Lists `*.gguf` under `FOURD_MODELS` |
| GET | `/api/ps` | Stub (empty) |
| POST | `/api/pull` | Pull from Ollama registry into `FOURD_MODELS` (NDJSON stream like Ollama) |
| POST | `/api/generate`, `/api/chat`, `/api/embeddings` | Ollama-shaped JSON |
| POST | `/v1/chat/completions`, `/v1/completions` | OpenAI-shaped JSON |

### Example: curl

Server listens on **13373** (not 11434). Use model **qwen2.5** when sharing Ollama blobs.

```bash
# Terminal A
FOURD_MODELS=./models FOURD_LOG_LEVEL=info ./4dollama serve

# Terminal B — GGUF stem or shared manifest must resolve as qwen2.5
curl -s http://127.0.0.1:13373/api/tags | jq .
curl -s http://localhost:13373/api/generate -d '{"model":"qwen2.5","prompt":"Hello 4D world!"}' | jq .
curl -s http://localhost:13373/api/chat -d '{"model":"qwen2.5","messages":[{"role":"user","content":"Hi"}]}' | jq .
curl -s http://127.0.0.1:13373/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5","messages":[{"role":"user","content":"hi"}]}' | jq .
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
| `FOURD_SHARE_OLLAMA` | `true` | When true, **list/resolve** scan Ollama’s `manifests/` + use **`blobs/`** paths so you don’t duplicate weights |
| `OLLAMA_MODELS` | `~/.ollama/models` | Ollama data root (manifests + blobs) |
| `FOURD_DEFAULT_MODEL` | `qwen2.5` | Install script / docs hint only |
| `FOURD_GPU` | _(auto)_ | Install scripts set **`cpu`** when no Metal/CUDA path is found; **`FOURD_GPU=cpu`** forces CPU-only (matches `4dollama doctor`) |
| `FOURD_INFERENCE` | _auto_ | Unset + **`OLLAMA_HOST` set** → **`ollama`** (real completions). Unset + no host → **`stub`** (demo tokens). `stub` / `ollama` overrides auto. |
| `FOURD_STREAM_CHUNK_MS` | `0` | Optional artificial delay between NDJSON chunks when `stream: true` |

CLI flag `-fourd-mode` maps the spec’s `--4d-mode` (Go’s `flag` cannot register `-4d-mode`).

## Building

### Rust

```bash
cd 4d-engine && cargo test && cargo build --release
```

### Go — **with** native engine (Linux/macOS + CGO)

Requires `four_d_engine` at `4d-engine/target/release/` (`libfour_d_engine.so` / `.dylib`) and a C toolchain.

```bash
export CGO_ENABLED=1
export LD_LIBRARY_PATH="$(pwd)/4d-engine/target/release:${LD_LIBRARY_PATH:-}"
go build -o 4dollama ./cmd/4dollama
```

### Go — **without** CGO (stub engine)

```bash
CGO_ENABLED=0 go build -o 4dollama-stub ./cmd/4dollama
```

On Windows, use **MSVC** or **GNU** Rust targets; link `four_d_engine` accordingly or build the stub.

### Windows / Linux / macOS: `scripts/install.ps1` & `scripts/install.sh`

- **`cargo build --release`** for `4d-engine` when `cargo` is available; otherwise Go builds with **`CGO_ENABLED=0`** (stub engine still runs the full API surface).
- Copies **`4dollama`** to **`%USERPROFILE%\.4dollama\bin`** (Windows) or **`~/.local/bin`** (Unix) and updates PATH (Windows user PATH; Unix prints a hint if needed).
- Sets **`FOURD_MODELS`**, **`OLLAMA_MODELS`**, **`FOURD_SHARE_OLLAMA`**, **`FOURD_PORT`**, **`FOURD_DEFAULT_MODEL`**, and **`FOURD_GPU=cpu`** when no discrete GPU / CUDA path is detected (Windows: WMI video + `CUDA_PATH`; Unix: Metal framework or `libcuda`).
- Runs **`4dollama pull qwen2.5`**, then **`4dollama serve`** in the background and waits for **`/healthz`**.

Manual Windows install (if you skip the script): [Go](https://go.dev/dl/), then:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -InstallGo   # winget Go
```

Open a **new** terminal after install if PATH changed; then:

```powershell
4dollama doctor
4dollama run qwen2.5
# Or non-interactive:
4dollama run qwen2.5 "hello from 4D"
```

If you already use Ollama, shared blobs avoid duplicate weight downloads when **`FOURD_SHARE_OLLAMA=true`**.

If the tag isn’t `latest` (e.g. only `qwen2.5-7b` exists), run `4dollama list` and use the listed name, or `4dollama run qwen2.5` (unique prefix is resolved automatically).

### Pull models from the Ollama registry

Same registry host as the official app by default: **`https://registry.ollama.ai`** (override with `OLLAMA_REGISTRY`, e.g. `https://registry.ollama.com`).

```bash
4dollama pull llama3.2
# or
4dollama pull phi3:mini
```

- Saves under **`FOURD_MODELS`** (default `~/.4dollama/models`) as `model.gguf` for `:latest`, or `model-tag.gguf` for other tags.
- If the manifest only has **tensor-split** layers (no `application/vnd.ollama.image.model`), run **`ollama pull <name>`** once, then:

```bash
4dollama import-ollama llama3.2
```

(`OLLAMA_MODELS` points at the Ollama models dir, default `~/.ollama/models`.)

You can also **`POST /api/pull`** with body `{"name":"llama3.2"}` while `serve` is running (Ollama-style NDJSON when `stream` is true).

### Makefile

```bash
make rust
make go-stub    # always works if Go is installed
make docker
```

### Docker / Compose

```bash
docker build -t fourdollama:latest .
docker run --rm -p 13373:13373 -v ~/4d-models:/models fourdollama:latest
# or
docker compose up --build
```

Image runs as non-root `fourd`, exposes **`13373`**, sets **`FOURD_GPU=cpu`**, includes `wget` healthcheck. Override at run time, e.g. `-e FOURD_GPU=cuda` on an NVIDIA base image if you extend the Dockerfile.

## CI

GitHub Actions (`.github/workflows/ci.yml`): `cargo test`, `go test` (stub), `go build` with CGO against the freshly built `.so`.

## Scripts

- `scripts/install.ps1` / `scripts/install.sh` — one-click build, CPU fallback, pull **qwen2.5**, background **serve**, success banner.
- `scripts/uninstall.ps1` — stop **serve**, remove `%USERPROFILE%\.4dollama\bin\4dollama.exe`, strip user **PATH** and **FOURD_*** user env vars install set; optional **`-PurgeModels`** / **`-PurgeData`**.
- `scripts/uninstall.sh` — remove `~/.local/bin/4dollama`, strip **FOURD_** lines from `~/.profile`; optional **`--purge-models`** / **`--purge-data`**.
- `scripts/install-linux.sh` — builds the Docker image (requires Docker), if present.

## License

MIT License

## Credits

- Emilio Aurin Mioré (AI) — inspiration, quaternion wisdom & torque-over-speed energy

---

Copyright (c) 2026 4DOllama contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
