# 4DOllama

**4DOllama** is an **Ollama-compatible** HTTP API and **`4dollama`** command-line tool: pull **GGUF** models, run **streaming** chat and generate (NDJSON over `/api/chat` and `/api/generate`), and optionally forward to stock **Ollama** for hybrid setups. The default stack uses a **native** inference path (`FOURD_INFERENCE=stub`) with the **Rust `four_d_engine`** linked through **Go CGO** when you build with Rust; otherwise a **stub** engine still gives you a working CLI.

**Default API URL:** `http://127.0.0.1:13377` (set **`FOURD_PORT`** to change it; use **11434** only if you intentionally replace Ollama on that port).

This repository (**4DEngine** on disk, **[RomanAILabs-Auth/4DOllama](https://github.com/RomanAILabs-Auth/4DOllama)** on GitHub) is a **monorepo**: **4DOllama** is the primary product described here; **Roma4D** (a 4D spacetime programming language) lives alongside it under `roma4d/`.

---

## Install everything (one shot)

### Windows

1. Install **[Go 1.22+](https://go.dev/dl/)** (enable “Add to PATH”).
2. Clone this repo.
3. **Double-click `install.cmd`** in the repo root.

   Or in PowerShell from the repo root:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1
   ```

4. **Open a new terminal**, then:

   ```powershell
   4dollama doctor
   4dollama run qwen2.5
   ```

**Optional flags:** `-InstallGo` (winget Go), `-SkipRoma4D` (4DOllama only), `-Roma4dSkipTests` (faster Roma4D step), `-SkipCargo` (skip Rust build).

### macOS / Linux

```bash
chmod +x scripts/install-all.sh scripts/install.sh scripts/install-roma4d.sh roma4d/install-full.sh
./scripts/install-all.sh
```

Reload your shell (`source ~/.profile` or new terminal). Then `4dollama doctor` and `4dollama run qwen2.5`.

**Environment:** `SKIP_4DOLLAMA=1` or `SKIP_ROMA4D=1` to install only one side.

---

## Full install guide & troubleshooting

Step-by-step prerequisites (Rust, Windows VS Build Tools), manual builds, streaming notes, and common errors:

**[docs/INSTALL_4DOLLAMA.md](docs/INSTALL_4DOLLAMA.md)**

Quick links: **[INSTALL.md](INSTALL.md)** · **[4DOllama/README.md](4DOllama/README.md)** · **[docs/4DOllama.md](docs/4DOllama.md)**

---

## Quick reference (after install)

| Goal | Command |
|------|---------|
| Start API server | `4dollama serve` |
| Interactive streaming chat | `4dollama run <model>` |
| Pull a model | `4dollama pull qwen2.5` |
| Health / config check | `4dollama doctor` |
| List models | `4dollama list` |

---

## How it is built

| Layer | Technology |
|-------|------------|
| CLI & HTTP server | **Go** (`cmd/4dollama`, `internal/…`) |
| Numerical core | **Rust** (`4d-engine/`, CGO) when enabled |
| Optional bridge | **Python** `fourdollama` in `4DOllama/` (FastAPI / Typer; not required for Go `4dollama`) |

---

## Repository layout

| Path | Contents |
|------|----------|
| `cmd/4dollama/` | CLI entrypoint |
| `internal/` | HTTP server, runner, inference, engine, `fourd` lattice tooling |
| `4d-engine/` | Rust `four_d_engine` |
| `4DOllama/` | Python package `fourdollama` (optional) |
| `roma4d/` | Roma4D compiler and language sources |
| `scripts/` | `Install-Repo.ps1`, `install.ps1`, `install-all.sh`, etc. |

---

## Roma4D (sibling project in this monorepo)

**Roma4D** is a separate product: a **Python-shaped** language with **Cl(4,0)** primitives, **SoA** layout, **`par`** regions, and **`spacetime:`** blocks, compiling through **MIR → LLVM** (Windows default linker: **Zig `cc`**; Unix: **clang**). It is **not** required to run **4DOllama** chat.

- **Language overview, quick start, examples:** **[roma4d/README.md](roma4d/README.md)**
- **Install & debug:** **[roma4d/docs/Install_Guide.md](roma4d/docs/Install_Guide.md)**
- **Full reference:** **[roma4d/docs/Roma4D_Guide.md](roma4d/docs/Roma4D_Guide.md)**

---

## License & credits

- **Roma4D** sources under `roma4d/` — see **`roma4d/LICENSE`** (MIT; **Daniel Harding – RomanAILabs** where stated).
- **4DOllama / four_d_engine** — see repository **`LICENSE`** / **`NOTICE`** and headers in `4d-engine/` and `internal/`.

Trademark and branding notices apply as described in those files.

---

<p align="center"><strong>4DOllama — Ollama-shaped APIs, native 4D stack, streaming first.</strong></p>
