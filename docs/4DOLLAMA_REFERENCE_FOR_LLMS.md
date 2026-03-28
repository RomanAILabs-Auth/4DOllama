# 4DOllama — reference for LLM agents and operators

This document is written so an **LLM or automation** can reason correctly about **what 4DOllama is**, **how to install and run it**, **what “4D” means in this codebase**, and **how to avoid common mistakes** (paths, inference modes, Python vs Go).

---

## What 4DOllama is (one paragraph)

**4DOllama** is a **Go** application (`cmd/4dollama`) that exposes an **Ollama-shaped** HTTP API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, …) and a **Cobra CLI** mirroring **Ollama’s verbs** (`serve`, `run`, `pull`, `list`, `create`, `stop`, `push`, `signin`, …). It integrates a **Rust** crate **`four_d_engine`** (via CGO when enabled) for **GGUF inspection** and **4D-flavored tensor paths** (quaternion RoPE, spacetime-style attention, 4D GEMM) in the **native** inference pipeline. The same repo may ship an **optional Python** package **`fourdollama`** under `4DOllama/` for a **FastAPI** bridge that shells out to **`r4d run`** on generated `kernel.r4d` files — that path is **separate** from the Go server’s GGUF decode.

---

## Is it “truly 4D architecture”?

**Honest answer:** It is **4D-aware software architecture**, not a claim that the host computer runs physics in four spatial dimensions.

- **Real in code:** Extra axes and **Cl(4,0)-inspired** / quaternion / **w-axis** operations in the **Rust** engine and **Go** runner; **RomanAI `.4dai`** carriers (JSON or ROMANAI4+safetensors) and **registry listing** for `.4dai` alongside **GGUF**.
- **Not implied:** That any particular benchmark beats all other stacks without measurement, or that “4D” replaces standard transformer math — the product **combines** usual autoregressive steps with **additional structured ops** where wired.

Use **`docs/ARCHITECTURE.md`** for the engineering contract and roadmap.

---

## Install (high level)

| Platform | Command / entry |
|----------|------------------|
| Windows | `install.cmd` or `scripts\Install-Repo.ps1` |
| Unix | `scripts/install-all.sh` or `scripts/install.sh` |

Binary target: **`%USERPROFILE%\.4dollama\bin\4dollama.exe`** (Windows) or **`~/.local/bin/4dollama`**.

After install: **`4dollama version`**, **`4dollama doctor`**, **`4dollama serve`**, **`4dollama run <model>`**.

Full narrative: **`docs/INSTALL_4DOLLAMA.md`**.

---

## CLI (Cobra) — commands an LLM should know

| Command | Role |
|---------|------|
| `serve` | HTTP API on **`FOURD_PORT`** (default **13377**). Flags: `--verbose`, `-p` / `--port`, `--host` |
| `run MODEL [prompt…]` | REPL (`>>> `) if TTY and no prompt; else one-shot or stdin |
| `pull MODEL` | Download **GGUF** from **registry.ollama.ai** into **`FOURD_MODELS`** |
| `list` | **`/api/tags`** — **GGUF** + shared Ollama library + **`.4dai`** under `FOURD_MODELS` |
| `create MODEL -f Modelfile` | **FROM** `.gguf` or `.4dai` (JSON merge for multiple shards); see below |
| `show`, `cp`, `rm`, `ps`, `stop` | Registry / API parity |
| `push`, `signin`, `signout` | Delegate to **`ollama`** on PATH when present |
| `launch` | Open **ollama.com** (or URL) in browser |
| `help`, `--help`, `-h` | Cobra help |
| Global | **`--fourd-mode`**, **`--version`** |

**Important:** Subcommand **`serve`** uses **`--host`**, not **`-h`**, because **`-h`** is reserved for help on the root command.

---

## `create` and Modelfile `FROM` paths (bulletproof rule)

Relative **`FROM ./shard.4dai`** is resolved against the **current working directory of the `4dollama create` process** (`os.Getwd()`), **not** the Modelfile’s directory and **not** any daemon cwd.

1. Resolve → **`filepath.Abs`**.
2. **`os.Stat`** each path before copy/merge.
3. On failure, print **`[4DOLLAMA FATAL] Shard not found! Searched absolute path: …`** and the **CWD** used.
4. Shards are also copied into **`$FOURD_MODELS/blobs/`** as **`{model}__{basename}`** for auditing.

Absolute `FROM` paths work as-is (cleaned + `Abs`).

---

## Python `fourdollama` and `r4d run` (kernel path bug fix)

The optional Python server writes jobs under **`$FOURDOLLAMA_DATA/work/<uuid>/kernel.r4d`** and runs:

`r4d run <path>` with **`cwd`** = that job directory.

**Bug (fixed):** If **`r4d`** joined **`cwd`** with a path that already contained **`work/<uuid>/`**, the runtime could look for **`…/work/<uuid>/work/<uuid>/kernel.r4d`**.

**Fix:** When **`kernel.r4d`** resolves **inside** **`cwd`**, the subprocess passes **only `kernel.r4d`** (basename) and a **resolved** **`cwd`**. Otherwise it passes the **full absolute** path.

Files: **`4DOllama/fourdollama/r4d_subprocess.py`**, **`engine.py`**, **`server.py`**.

---

## Environment variables (minimal set)

| Variable | Meaning |
|----------|---------|
| `FOURD_PORT` / `FOURD_HOST` | Bind |
| `FOURD_MODELS` | Model dir (default `~/.4dollama/models`) |
| `FOURD_INFERENCE` | `stub` / `fourd` / `native` vs `ollama` hybrid |
| `OLLAMA_HOST` | Upstream Ollama for hybrid / benchmarks |
| `FOURD_SHARE_OLLAMA` | List/pull reuse **`~/.ollama/models`** |
| `FOURDOLLAMA_DATA` | Python bridge data dir (default `~/.fourdollama`) |
| `FOURDOLLAMA_R4D` | Path to **`r4d`** executable for Python bridge |

---

## How an LLM should answer “how do I chat?”

1. Ensure a model exists: **`4dollama pull qwen2.5`** (or copy GGUF into **`FOURD_MODELS`**).
2. **`4dollama serve`** (or rely on **`4dollama run`** auto-starting serve).
3. **`4dollama run qwen2.5`** for **`>>> `** REPL, or **`4dollama run qwen2.5 "hello"`** for one-shot.

For **OpenAI-style** clients, use **`POST /v1/chat/completions`** on the same port.

---

## Related docs

- **`docs/4DOllama.md`** — Product guide, API table, build matrix.
- **`docs/ARCHITECTURE.md`** — Layers, honesty, roadmap.
- **`docs/INSTALL_4DOLLAMA.md`** — Step-by-step install and troubleshooting.
- **`4DOllama/README.md`** — Optional Python **`fourdollama`** package.

---

*RomanAILabs / 4DEngine — keep claims aligned with measured behavior.*
