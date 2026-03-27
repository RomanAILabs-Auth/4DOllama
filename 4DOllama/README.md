# 4DOllama

Ollama-shaped REST + CLI over **Roma4D** (`r4d`). Default bind: **127.0.0.1:13377**.

## Native 4D engine (`fourd`) — Cl(4,0) + 4D lattice

The **Go** binary (`cmd/4dollama`) embeds a numerical **4DOllama** substrate:

| Component | Path | Role |
|-----------|------|------|
| **Cl(4,0)** | `internal/fourd/clifford` | 16-blade multivectors, geometric product, rotors, isoclinic rotor composition |
| **4D lattice** | `internal/fourd/lattice4` | Periodic grid, Laplacian, **heat/diffusion** step (stable demo), leapfrog **wave** (needs CFL tuning) |
| **Q-tensor coupling** | `internal/fourd/coupling` | \(\|QK^\top\|_F\) → source field; lattice → logit-bias stub |
| **Hodge helpers** | `internal/fourd/hodge` | Mean removal; DEC / Betti are **roadmap** |
| **Orchestrator** | `internal/fourd/orchestrator` | Bounded stepping loop with injection cadence |

**Run (from monorepo root next to `go.mod`):**

```bash
go build -o 4dollama ./cmd/4dollama
./4dollama fourd ga-demo
./4dollama fourd lattice -steps 80 -kappa 0.002 -inject-every 8
```

**Launcher:** `scripts/Launch-4DOllama.ps1` (builds and runs the above).

**Full math + phased roadmap:** [docs/NATIVE_4D_ENGINE.md](docs/NATIVE_4D_ENGINE.md).

**RQ4D** (RomanAILabs Go quantum / core repo) is a **sister runtime**; macro-DAG execution lives under `RQ4D/internal/core`. This package stays the **4dollama** integration point for HTTP + Roma4D tooling.

### Ollama-compatible CLI & API (drop-in)

Same command verbs as **Ollama**: `serve`, `run`, `pull`, `list`, `ps`, `show`, `cp`, `rm`, `version`, …

**Routes** (same paths as Ollama): `/api/version`, `/api/tags`, `/api/pull`, `/api/generate`, `/api/chat`, `/api/ps`, `/v1/chat/completions`, …

**Drop-in replacement** (stop Ollama first to free the port):

```powershell
$env:FOURD_DROPIN = "1"
4dollama serve
# listens on http://127.0.0.1:11434
```

Or explicitly:

```powershell
4dollama serve -h 127.0.0.1 -p 11434
```

**Native inference path:** `FOURD_INFERENCE=stub` (default) runs **GGUF tokenizer + four_d_engine** autoregressive steps. Each step builds **RoPE** and **SpacetimeAttention4D**; those quads pack into **Q/K-shaped** buffers; **||QKᵀ||_F** injects **cognitive gravity** into a **4D torus** field and **feeds back** as a **logit bias** (`internal/inference/coupling_lattice.go`). Tune strength with **`FOURD_LATTICE_KAPPA`**.

```powershell
4dollama pull qwen2.5
4dollama run qwen2.5 "Hello from native 4D lattice"
```

## Next to Ollama (no port clash)

| | **Ollama** | **4DOllama (this package)** |
|--|------------|-----------------------------|
| Default API | `http://127.0.0.1:11434` | `http://127.0.0.1:13377` |
| CLI | `ollama` | `4dollama` / `4dollam` / `python -m fourdollama` |

Keep **`FOURDOLLAMA_PORT` unset** (or explicitly `13377`) so both can run at once. Only change the port if you intentionally replace Ollama on the same socket.

### Wrong UI (pink `4dollama · … · chat`, duplicate “Message… Ollama-style”)

That screen is from an **old Go `4dollama.exe`** (Bubble Tea), which is **removed in source**. You are still running a stale binary (often `%USERPROFILE%\go\bin\4dollama.exe` or an old copy on `PATH`).

1. In PowerShell: `Get-Command 4dollama -All` (or `where.exe 4dollama`).
2. **Prefer this repo’s Python CLI** (plain `>>>` only, no dashboard):  
   `python -m fourdollama run qwen2.5`  
   after `pip install -e` from `4DOllama/`.
3. Or **rebuild** Go from an updated clone: `go build -o 4dollama.exe ./cmd/4dollama` and replace the exe you actually run.

```powershell
cd 4DOllama
# Use the SAME Python as `quantum_win` (or your venv):
python -m pip install -e .
# If `4dollam` / `4dollama` is still "not recognized", Scripts is not on PATH:
$env:Path = "$(python -c 'import sys; print(sys.prefix)')\Scripts;$env:Path"
# Or always (no console script needed):
python -m fourdollama run qwen2.5

$env:R4D_PKG_ROOT="C:\path\to\roma4d"
$env:FOURDOLLAMA_R4D="C:\path\to\r4d.exe"
4dollama serve
```

**quantum_win:** from repo root, after `quantum_win`:

`.\4DOllama\scripts\Install-IntoCurrentVenv.ps1`

Env: `FOURDOLLAMA_HOST`, `FOURDOLLAMA_PORT`, `FOURDOLLAMA_R4D`, `R4D_PKG_ROOT`, `FOURDOLLAMA_DATA`, `FOURDOLLAMA_R4D_TIMEOUT`.

**Go `4dollama` (repo `cmd/4dollama`):** interactive `run` is **only** the plain Ollama-style `>>>` line REPL (bubble UI removed). Rebuild the binary after updates.
