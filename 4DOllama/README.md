# 4DOllama

Ollama-shaped REST + CLI over **Roma4D** (`r4d`). Default bind: **127.0.0.1:13377**.

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
