# 4DOllama

Ollama-shaped REST + CLI over **Roma4D** (`r4d`). Default bind: **127.0.0.1:13377**.

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
