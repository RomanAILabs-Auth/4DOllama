# Install this repository

This repo (**4DEngine** / **4DOllama** on GitHub) ships two major tools:

1. **4DOllama** — `4dollama` CLI and Ollama-compatible HTTP API (default **http://127.0.0.1:13377**), streaming chat/generate.
2. **Roma4D** — the `.r4d` language and **`r4d`** compiler under `roma4d/`.

---

## Fastest start

| OS | What to run |
|----|-------------|
| **Windows** | Double-click **`install.cmd`** in the repo root. |
| **Windows (PowerShell)** | `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1` |
| **macOS / Linux** | `chmod +x scripts/install-all.sh && ./scripts/install-all.sh` |

Then **open a new terminal** and run **`4dollama doctor`**.

---

## Full guide (step-by-step, troubleshooting)

**[docs/INSTALL_4DOLLAMA.md](docs/INSTALL_4DOLLAMA.md)**

---

## Product docs

| Topic | Link |
|-------|------|
| 4DOllama overview | [4DOllama/README.md](4DOllama/README.md) |
| API, Docker, env | [docs/4DOllama.md](docs/4DOllama.md) |
| Roma4D language | [roma4d/README.md](roma4d/README.md) |
| Roma4D install only | [roma4d/docs/Install_Guide.md](roma4d/docs/Install_Guide.md) |

---

## Script reference

| Script | Purpose |
|--------|---------|
| `install.cmd` | Windows double-click → `Install-Repo.ps1` |
| `scripts/Install-Repo.ps1` | **4DOllama** (`install.ps1`) + **Roma4D** (`Install-Roma4d.ps1`) |
| `scripts/install.ps1` | 4DOllama only (Windows) |
| `scripts/install-all.sh` | 4DOllama + Roma4D (Unix) |
| `scripts/install.sh` | 4DOllama only (Unix) |
| `scripts/Install-Roma4d.ps1` | Roma4D only (Windows) |
| `scripts/install-roma4d.sh` | Roma4D only (Unix) |
