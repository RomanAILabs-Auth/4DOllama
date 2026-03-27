# 4DOllama — full install guide

This guide is for the **RomanAILabs-Auth/4DOllama** monorepo (**4DEngine**): **4DOllama** (Ollama-compatible server + `4dollama` CLI) and, in the same tree, **Roma4D** (the `.r4d` language compiler). If you only care about chat/GGUF/APIs, you only need the **4DOllama** steps.

---

## Easiest path (recommended)

### Windows

1. Install **[Go 1.22+](https://go.dev/dl/)** (64-bit). During setup, allow **“Add to PATH”**.
2. **Clone** the repo and open the folder in Explorer.
3. **Double-click** **`install.cmd`** in the repo root.

   Or open **PowerShell** *as your normal user* (not necessarily admin), `cd` to the repo root, then:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1
   ```

4. **Close and reopen** your terminal (PATH and user environment variables refresh).
5. Verify:

   ```powershell
   4dollama version
   4dollama doctor
   ```

6. Chat (streaming REPL, same idea as `ollama run`):

   ```powershell
   4dollama run qwen2.5
   ```

**Optional:** If Go is missing, run:

```powershell
.\scripts\Install-Repo.ps1 -InstallGo
```

(requires **winget**). Or install Go manually from https://go.dev/dl/ .

**Rust / native engine:** The installer runs `cargo build --release` on `4d-engine` when `cargo` is on PATH. If that fails (common on Windows without **Visual Studio Build Tools + C++**), the script **falls back** to `CGO_ENABLED=0` — you still get a working `4dollama` binary; the native Rust engine is stubbed until you fix the Rust toolchain.

**Faster Roma4D step (skip compiler tests):**

```powershell
.\scripts\Install-Repo.ps1 -Roma4dSkipTests
```

**Only 4DOllama (skip Roma4D):**

```powershell
.\scripts\Install-Repo.ps1 -SkipRoma4D
```

---

### macOS / Linux

1. Install **Go** and **git**. On Ubuntu/Debian: `sudo apt install golang-go git build-essential` (and **`clang`** for Roma4D — see `roma4d/docs/Install_Guide.md`).
2. Clone and enter the repo root.
3. Run:

   ```bash
   chmod +x scripts/install-all.sh scripts/install.sh scripts/install-roma4d.sh roma4d/install-full.sh
   ./scripts/install-all.sh
   ```

4. Ensure **`~/.local/bin`** is on your `PATH` (the script prints a hint if not). Reload:

   ```bash
   source ~/.profile   # or open a new terminal
   ```

5. Verify:

   ```bash
   4dollama version
   4dollama doctor
   4dollama run qwen2.5
   ```

**Only 4DOllama:**

```bash
SKIP_ROMA4D=1 ./scripts/install-all.sh
# or
./scripts/install.sh
```

**Only Roma4D:**

```bash
SKIP_4DOLLAMA=1 ./scripts/install-all.sh
```

---

## What the installer does (4DOllama)

| Step | Action |
|------|--------|
| 1 | Builds **`four_d_engine`** with Cargo when possible (`4d-engine/Cargo.toml`). |
| 2 | Builds **`4dollama`** from **`cmd/4dollama`** (CGO on if Rust lib exists). |
| 3 | Copies the binary to **`%USERPROFILE%\.4dollama\bin`** (Windows) or **`~/.local/bin`** (Unix). |
| 4 | Prepends that directory to your **user PATH** (Windows) or tells you to export PATH (Unix). |
| 5 | Sets user env vars: **`FOURD_MODELS`**, **`FOURD_PORT=13377`**, **`FOURD_INFERENCE=stub`**, **`OLLAMA_HOST`**, **`FOURD_SHARE_OLLAMA`**, etc. |
| 6 | Best-effort **`4dollama pull qwen2.5`** (needs network). |
| 7 | Starts **`4dollama serve`** in the background (health check on **http://127.0.0.1:13377**). |

Default **inference** is **native stub** (`FOURD_INFERENCE=stub`). To forward completions to stock Ollama instead, set **`FOURD_INFERENCE=ollama`** and **`OLLAMA_HOST=http://127.0.0.1:11434`** after Ollama is running.

---

## Manual install (no scripts)

From repo root:

```bash
# Optional: Rust engine
cargo build --release --manifest-path 4d-engine/Cargo.toml
export CGO_ENABLED=1   # or 0 if no Rust lib

go build -o 4dollama ./cmd/4dollama
./4dollama serve
```

In another terminal:

```bash
./4dollama pull qwen2.5
./4dollama run qwen2.5
```

---

## Streaming and API

- **CLI:** `4dollama run <model>` uses a plain **`>>> `** line interface; assistant text **streams** over **`POST /api/chat`** with `stream: true`.
- **HTTP:** NDJSON on **`/api/chat`** and **`/api/generate`** when streaming — see **[4DOllama.md](4DOllama.md)** and **[../4DOllama/README.md](../4DOllama/README.md)**.

---

## Common problems

| Symptom | What to do |
|---------|------------|
| **`4dollama` is not recognized** | New terminal after install; check `where 4dollama` / `which 4dollama`. Ensure `~/.4dollama/bin` or `~/.local/bin` is on PATH. |
| **Pink / bubble UI, duplicate prompts** | Old binary on PATH. Rebuild and replace, or use `%USERPROFILE%\.4dollama\bin\4dollama.exe` first on PATH. |
| **Connection refused on 13377** | Run `4dollama serve` (foreground) and read stderr. |
| **Pull fails** | Network/firewall; registry.ollama.ai must be reachable. |
| **CGO / Rust link errors on Windows** | Install **VS Build Tools** with **Desktop development with C++**, or use `-SkipCargo` and accept stub engine until fixed. |

---

## Optional: Python bridge (`fourdollama`)

For the FastAPI / Typer helper under **`4DOllama/`** (Roma4D integration):

```bash
cd 4DOllama
pip install -e .
python -m fourdollama serve
```

This is **not** required for the main Go **`4dollama`** server.

---

## Roma4D only

The **language** and **`r4d`** compiler live under **`roma4d/`**. Full Roma4D install and troubleshooting:

- **[../roma4d/README.md](../roma4d/README.md)**
- **[../roma4d/docs/Install_Guide.md](../roma4d/docs/Install_Guide.md)**

---

## See also

| Document | Purpose |
|----------|---------|
| [../4DOllama/README.md](../4DOllama/README.md) | Product overview, architecture, streaming |
| [4DOllama.md](4DOllama.md) | API, env vars, Docker, curl examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Monorepo architecture |
