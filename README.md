<div align="center">

# 4DOllama

### *The Ollama-compatible runtime — reimagined through native 4D spacetime inference.*

[![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?style=for-the-badge&logo=go&logoColor=white)](https://go.dev/dl/)
[![Rust](https://img.shields.io/badge/Rust-four__d__engine-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge)]()

**Drop-in CLI & HTTP API · Streaming NDJSON · GGUF registry · Cl(4,0) geometric core · Lattice ↔ logit coupling**

[Installation](#-installation) · [Quick Start](#-quick-start) · [Architecture](#-architecture-overview) · [Math](#-mathematical-foundation) · [Roma4D & RQ4D](#-roma4d-the-science-behind-the-stack) · [Docs](#-documentation-index)

---

**Copyright © 2026 Daniel Harding - RomanAILabs. All Rights Reserved.**

*Software is released under the [Apache License 2.0](#-license) (see [LICENSE](LICENSE)).  
RomanAILabs names, logos, and branding are separate from the license grant.*

---

</div>

<br/>

## Table of contents

| | |
|--|--|
| [Why 4DOllama?](#-why-4dollama) | [Features](#-feature-matrix) |
| [Architecture](#-architecture-overview) | [Mathematical foundation](#-mathematical-foundation) |
| [Roma4D & RQ4D](#-roma4d-the-science-behind-the-stack) | [Installation](#-installation) |
| [Quick start](#-quick-start) | [Usage examples](#-usage-examples) |
| [Advanced](#-advanced-features) | [Configuration](#-configuration-reference) |
| [API surface](#-api-surface-ollama-compatible) | [Troubleshooting](#-troubleshooting) |
| [Roadmap](#-roadmap) | [Contributing](#-contributing) |
| [License](#-license) | [Contact](#-contact) |

<br/>

## Why 4DOllama?

**4DOllama** is a **production-oriented**, **Ollama-shaped** command-line tool and HTTP server. It speaks the same verbs developers already know — `pull`, `run`, `serve`, `list`, `ps`, `show`, and more — and exposes the familiar REST routes (`/api/chat`, `/api/generate`, `/api/tags`, …).

What makes it different is the **native 4D inference spine**: quaternion-style rotary embeddings, **spacetime attention** over token quads, optional **lattice dynamics** with **Q–tensor style coupling**, and feedback into the decode loop as **logit bias** — all within the **RomanAILabs 4D ecosystem** (Rust **`four_d_engine`**, Go **`internal/fourd`**, and the **Roma4D** / **RQ4D** research stack living alongside this repo).

> **4DOllama first.** This README is the front door for the GitHub project. Roma4D and RQ4D are the **language and research engines** that inform the geometry and roadmap — they are not a prerequisite to run `4dollama run` on day one.

<br/>

## Feature matrix

| Capability | 4DOllama |
|------------|----------|
| **Ollama CLI verbs** | `serve`, `run`, `chat`, `pull`, `list`, `ps`, `show`, `cp`, `rm`, `version`, … |
| **Streaming** | ✅ NDJSON on `/api/chat` & `/api/generate` with flush-per-chunk |
| **Interactive REPL** | ✅ Plain `>>> ` line mode (Ollama-style; no full-screen bubble TUI) |
| **GGUF** | ✅ Pull, resolve, optional reuse of `~/.ollama/models` blobs |
| **Default port** | **13377** — runs **beside** Ollama on **11434** |
| **Default inference** | ✅ Forwards to **local Ollama** at `http://127.0.0.1:11434` for llama.cpp text (`FOURD_INFERENCE=stub` to use native decode only) |
| **Silent RQ4D** | ✅ Quantum lattice steps + Cl(4,0) lattice coupling on every request (no CLI flag) |
| **OpenAI-shaped routes** | ✅ `/v1/chat/completions`, `/v1/completions` (subset) |
| **Health & metrics** | ✅ `/healthz`, `/livez`, `/metrics` |
| **Docker** | ✅ `Dockerfile` + `docker-compose.yml` in repo |

<br/>

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Clients & tools                                │
│   4dollama CLI · curl · OpenAI SDKs · LangChain · custom HTTP            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP / NDJSON streams
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  internal/httpserver  —  chi router, middleware, streaming writers       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  internal/runner  —  resolve model, build inference context, stream      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  internal/inference  —  native stub path · optional Ollama forward       │
│  internal/inference/coupling_lattice.go — lattice ↔ logit bias           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  internal/engine  —  CGO ↔ Rust four_d_engine (or stub without CGO)     │
│  internal/fourd   —  Cl(4,0) helpers, lattice4, orchestrator demos        │
└─────────────────────────────────────────────────────────────────────────┘
```

| Component | Language | Role |
|-----------|----------|------|
| **`cmd/4dollama`** | Go | CLI entry, process orchestration |
| **`internal/httpserver`** | Go | Ollama-compatible HTTP API |
| **`internal/runner`** | Go | Chat/generate orchestration, streaming callbacks |
| **`4d-engine/`** | Rust | `four_d_engine`: RoPE, attention, GEMM, GGUF FFI |
| **`internal/fourd/`** | Go | Clifford/lattice demos, `4dollama fourd` subcommand |
| **`4DOllama/` (optional)** | Python | `fourdollama` Typer/FastAPI bridge for Roma4D workflows |

<br/>

## Mathematical foundation

### Clifford algebra Cl(4,0)

**Cl(4,0)** is the **Euclidean Clifford algebra in four dimensions** (signature **++++**). It unifies scalars, vectors, bivectors, trivectors, and pseudoscalars into a single **graded** object — a natural algebra for **rotations**, **boosts in higher-dimensional embeddings**, and **structured attention** over 4-vectors.

In 4DOllama’s stack, **token quads** (four floats per logical feature block) participate in **quaternion-style rotary mixing** and **4D attention** before coupling signals are injected into a **lattice** field.

### Rotors & spacetime mixing

**Rotors** are even-grade elements that encode **plane rotations**. On the engine path, rotary structure appears in **RoPE-like** transforms adapted to **quaternion** semantics over embedding quads — not merely complex phase on pairs.

### Q–tensor coupling & “cognitive gravity”

A running theme in RomanAILabs research is treating **attention geometry** as a physical field:

- Tensors shaped like **Q/K** pathways yield coupling strengths (e.g. **Frobenius norms** of structured products).
- Those strengths **inject** into a **4D lattice** (diffusion / wave-style updates in demos).
- The lattice state **projects back** to the decoder as **logit bias**, closing a **bidirectional** loop: *inference shapes the field; the field nudges the next token.*

Tune coupling with environment variables such as **`FOURD_LATTICE_KAPPA`** (see [Configuration](#-configuration-reference)).

### Hodge & DEC (roadmap)

**Hodge theory** and **discrete exterior calculus (DEC)** provide a language for **harmonic decomposition** of fields on meshes. Parts of this appear as **roadmap** items in `internal/fourd` (mean removal today; richer Hodge/Betti story evolving).

> This section is a **conceptual map** for researchers. Production behavior is defined by the code paths in `4d-engine` and `internal/inference` for each release.

<br/>

## Roma4D — the science behind the stack

**Roma4D** is RomanAILabs’ **4D spacetime programming language**: Python-clear syntax, **SoA** layout, **`par`** regions, and **`spacetime:`** blocks that compile through **MIR → LLVM** — a **compiler research vehicle** for how **geometry + parallelism** should look in source code.

**RQ4D** (in the broader RomanAILabs ecosystem) explores **quantum / lattice / executor** formulations of 4D workloads — a **sister runtime** direction that informs how we think about **scheduling**, **tensor networks**, and **macro-DAG** execution at scale.

**Relationship to 4DOllama:**

| Layer | How it connects |
|-------|-----------------|
| **4DOllama (this product)** | Ships the **Ollama-compatible** server and CLI most users install. |
| **`four_d_engine` (Rust)** | The **numerical core** linked from Go (CGO) for RoPE, attention, sampling. |
| **Roma4D (`roma4d/`)** | Lives in the **same monorepo**; optional for GGUF chat, essential for **`.r4d` language** work and native codegen experiments. |
| **RQ4D** | Research / executor stack; **conceptual and engineering lineage** for geometric runtimes. |

If your goal is **chat + API parity**, start with **`4dollama`**. If your goal is **language design + LLVM-native 4D programs**, live in **`roma4d/`** and read **`roma4d/docs/Roma4D_Guide.md`**.

<br/>

## Installation

### One-shot (recommended)

#### Windows

1. Install **[Go 1.22+](https://go.dev/dl/)** and allow **Add to PATH**.
2. Clone the repository.
3. **Double-click `install.cmd`** in the repo root.

   Or run in **PowerShell**:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1
   ```

4. **Close and reopen** your terminal (user `PATH` and env vars refresh).
5. Verify: `4dollama doctor` and `4dollama version`.

#### macOS / Linux

```bash
chmod +x scripts/install-all.sh scripts/install.sh scripts/install-roma4d.sh roma4d/install-full.sh
./scripts/install-all.sh
```

Reload your shell (`source ~/.profile` or open a new terminal). Ensure `~/.local/bin` (or your install prefix) is on `PATH`.

### Prerequisites (detailed)

| Prerequisite | Why | Notes |
|--------------|-----|-------|
| **Go 1.22+** | Build `4dollama` | Required |
| **Git** | Clone / update | Required |
| **Rust + Cargo** | Build `four_d_engine` | Optional but recommended for full native engine |
| **Windows: VS Build Tools (C++)** | Link Rust `cdylib` for CGO | If missing, installer falls back to `CGO_ENABLED=0` stub |
| **Zig or clang** | Roma4D `r4d` linker | Only if you compile **Roma4D** (see `roma4d/docs/Install_Guide.md`) |
| **Python 3.10+** | Optional `fourdollama` package | `pip install -e 4DOllama/` |

### Installer flags (Windows)

| Flag | Effect |
|------|--------|
| `-InstallGo` | Attempt **winget** install of Go |
| `-SkipRoma4D` | **4DOllama only** (skip compiler install) |
| `-Skip4DOllama` | **Roma4D only** |
| `-SkipCargo` | Skip Rust build (faster; stub engine) |
| `-Roma4dSkipTests` | Faster Roma4D step |

### Unix environment overrides

| Variable | Meaning |
|----------|---------|
| `SKIP_4DOLLAMA=1` | Only Roma4D path in `install-all.sh` |
| `SKIP_ROMA4D=1` | Only 4DOllama path |

### Full prose guide

📘 **[docs/INSTALL_4DOLLAMA.md](docs/INSTALL_4DOLLAMA.md)** — troubleshooting, manual builds, streaming notes, Docker.

📘 **[INSTALL.md](INSTALL.md)** — script index and quick links.

<br/>

## Quick start

```bash
# Terminal A — start the API (default http://127.0.0.1:13377)
4dollama serve

# Terminal B — ensure stock Ollama is running (llama.cpp on :11434), then:
4dollama pull phi3
4dollama run phi3
# Same as Ollama’s chat entrypoint:
4dollama chat phi3
```

### RomanAI and natural chat

**Replies match what you expect from Ollama** because completions are **forwarded to your local Ollama** by default (`OLLAMA_HOST`, default `http://127.0.0.1:11434`). Keep the **same model names** in both tools (`phi3`, `qwen2.5`, …). The default **RomanAI** system prompt keeps answers in **clear, friendly English** without surfacing geometry or stack jargon. **RQ4D + Cl(4,0) lattice work runs in-process on every request** and does not change the transcript.

Engine diagnostics stay **off the conversation** unless **`4dollama serve --verbose`** or **`FOURD_LOG_LEVEL=debug`**. To use **native stub decode** only (no Ollama), set **`FOURD_INFERENCE=stub`**. To skip the Ollama pre-pass when both are configured, set **`FOURD_4DAI_OLLAMA=0`**. **`FOURD_ALLOW_STUB_GARBAGE=1`** enables raw stub sampling (not recommended for chat).

**One-liner health check:**

```bash
curl -s http://127.0.0.1:13377/healthz
```

**Doctor (GPU hint, paths, listen URL):**

```bash
4dollama doctor
```

<br/>

## Usage examples

### Streaming generate (HTTP)

```bash
curl -sN http://127.0.0.1:13377/api/generate -H "Content-Type: application/json" -d '{
  "model": "qwen2.5",
  "prompt": "Explain quaternion rotations in one paragraph.",
  "stream": true
}'
```

### Streaming chat (HTTP)

```bash
curl -sN http://127.0.0.1:13377/api/chat -H "Content-Type: application/json" -d '{
  "model": "qwen2.5",
  "messages": [{"role":"user","content":"Hello from 4DOllama!"}],
  "stream": true
}'
```

### List local models

```bash
4dollama list
```

### 4D lattice demos (Go subcommand)

```bash
4dollama fourd ga-demo
4dollama fourd lattice -steps 80 -kappa 0.002 -inject-every 8
```

### Docker

```bash
docker compose up --build
# API: http://127.0.0.1:13377 (see docker-compose.yml)
```

<br/>

## Advanced features

| Topic | How |
|-------|-----|
| **Run beside Ollama** | Default `FOURD_PORT=13377`; keep Ollama on `11434`. |
| **Bind Ollama’s port** | Stop Ollama; `4dollama serve -h 127.0.0.1 -p 11434`. |
| **Hybrid inference** | Default: `FOURD_INFERENCE=ollama` + `OLLAMA_HOST=http://127.0.0.1:11434`. |
| **Lattice coupling strength** | `FOURD_LATTICE_KAPPA` (see source / docs). |
| **Smoother streaming UX** | `FOURD_STREAM_CHUNK_MS` (milliseconds between NDJSON chunks). |
| **Verbose engine logs** | `4dollama serve -verbose` or `FOURD_LOG_LEVEL=debug`. |
| **RomanAI chat tone** | Default system prompt is natural English; technical appendix only with `-verbose` / debug. |
| **Prefer Ollama for text** | `FOURD_4DAI_OLLAMA` / `FOURD_TRY_OLLAMA_FIRST` (on when `OLLAMA_HOST` set). |

<br/>

## Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `FOURD_HOST` | `0.0.0.0` | Bind address |
| `FOURD_PORT` | `13377` | HTTP port |
| `FOURD_MODELS` | `~/.4dollama/models` | GGUF tree |
| `FOURD_INFERENCE` | `ollama` | Upstream llama.cpp via Ollama; `stub` for native decode only |
| `FOURD_4DAI_OLLAMA` / `FOURD_TRY_OLLAMA_FIRST` | on if `OLLAMA_HOST` non-empty | Try Ollama before stub when inference is stub |
| `FOURD_4DAI_OLLAMA_MODEL` | _(unset)_ | Ollama model tag if it differs from the 4dollama name |
| `FOURD_ALLOW_STUB_GARBAGE` | `0` | Set `1` to allow raw stub token streams (not for normal chat) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Upstream Ollama API (set empty only if you use stub-only and avoid forward) |
| `FOURD_SHARE_OLLAMA` | `true` | Reuse Ollama blobs when present |
| `OLLAMA_MODELS` | `~/.ollama/models` | Ollama data root |
| `FOURD_LOG_LEVEL` | `warn` | `debug` / `info` / `warn` / `error` |
| `FOURD_DROPIN` | `0` | If `1`, pins loopback when `FOURD_HOST` unset |

Full tables: **[docs/4DOllama.md](docs/4DOllama.md)** · **LLM/agent reference:** **[docs/4DOLLAMA_REFERENCE_FOR_LLMS.md](docs/4DOLLAMA_REFERENCE_FOR_LLMS.md)** · **Architecture / “4D” honesty:** **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

<br/>

## API surface (Ollama-compatible)

| Method | Path | Notes |
|--------|------|------|
| GET | `/api/version` | Server version |
| GET | `/api/tags` | Model list |
| POST | `/api/pull` | Registry pull (NDJSON stream) |
| POST | `/api/generate` | Generate (+ stream) |
| POST | `/api/chat` | Chat (+ stream) |
| POST | `/api/embeddings` | Embeddings (subset) |
| GET | `/healthz`, `/livez` | Probes |
| GET | `/metrics` | Prometheus text |

OpenAI-shaped: `/v1/chat/completions`, `/v1/completions` (compatibility subset).

<br/>

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `4dollama` not found | New terminal after install; `where 4dollama` / `which 4dollama`. |
| Old bubble / pink UI | Stale binary on PATH — rebuild; prefer `%USERPROFILE%\.4dollama\bin\4dollama.exe` first. |
| `Connection refused` | Run `4dollama serve`; check `FOURD_PORT`. |
| Rust / CGO link errors (Windows) | Install **VS Build Tools** + **C++** workload, or `-SkipCargo` for stub. |
| Pull failures | Check firewall / `registry.ollama.ai` reachability. |
| Hybrid mode errors | Ensure `OLLAMA_HOST` points to a running Ollama. |

More: **[docs/INSTALL_4DOLLAMA.md](docs/INSTALL_4DOLLAMA.md#common-problems)**.

<br/>

## Roadmap

- [ ] Richer **Hodge / DEC** tooling in `internal/fourd`
- [ ] Deeper **GPU** paths in `four_d_engine` (beyond scaffolding)
- [ ] Expanded **OpenAI** compatibility matrix
- [ ] **Signed** release binaries + SBOM
- [x] **RQ4D** quantum sidecar (`RQ4D/rq4dsidecar`) on every inference (silent)
- [ ] Continued **Roma4D** ↔ **4DOllama** demo bridges

*Roadmap items are aspirational until shipped in a tagged release.*

<br/>

## Contributing

We welcome issues and PRs that **respect the architecture** (Go server + Rust engine + optional Python bridge).

1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feat/amazing-4d-idea`).
3. Run **`go test ./...`** from the repo root; add tests for new behavior.
4. Match existing **style** (focused diffs, no unrelated churn).
5. Open a **pull request** with a clear description and motivation.

For **Roma4D compiler** changes, work primarily under `roma4d/` and follow **`roma4d/docs/Install_Guide.md`**.

<br/>

## License

**Copyright © 2026 Daniel Harding - RomanAILabs. All Rights Reserved.**

The software is licensed under the **Apache License, Version 2.0**. RomanAILabs **names, logos, and branding** are **not** licensed for use as your product identity without permission — the Apache 2.0 grant covers the **code**, not trademark rights.

### Apache License 2.0

The full license text is in the repository root **[LICENSE](LICENSE)** file. A short SPDX-style summary:

```
Copyright 2026 Daniel Harding - RomanAILabs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Component-specific copies of the same license may appear in subdirectories (e.g. `roma4d/LICENSE`, `RQ4D/LICENSE`, `RomanAI/LICENSE`). When in doubt, retain **all** copyright headers present in files you redistribute.

<br/>

## Contact

- **Repository:** [github.com/RomanAILabs-Auth/4DOllama](https://github.com/RomanAILabs-Auth/4DOllama)
- **Issues:** use GitHub **Issues** for bugs, features, and install support threads.
- **Project:** **RomanAILabs** — *Daniel Harding*
- **Email:** [romanailabs@gmail.com](mailto:romanailabs@gmail.com) · [daniel@romanailabs.com](mailto:daniel@romanailabs.com)

---

<div align="center">

**Built with conviction at the intersection of inference, geometry, and systems.**

🌀 **4DOllama** — *Ollama-shaped. Four-dimensional at the core.*

</div>
