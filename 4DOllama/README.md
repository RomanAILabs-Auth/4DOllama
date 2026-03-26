# 4DOllama

Ollama-shaped REST + CLI over **Roma4D** (`r4d`). Default bind: **127.0.0.1:13377**.

```powershell
cd 4DOllama
pip install -e .
$env:R4D_PKG_ROOT="C:\path\to\roma4d"
$env:FOURDOLLAMA_R4D="C:\path\to\r4d.exe"
4dollama run qwen2.5
# one-shot: 4dollama run qwen2.5 hello there
# typo alias: 4dollam run llama3
4dollama serve
```

Env: `FOURDOLLAMA_HOST`, `FOURDOLLAMA_PORT`, `FOURDOLLAMA_R4D`, `R4D_PKG_ROOT`, `FOURDOLLAMA_DATA`, `FOURDOLLAMA_R4D_TIMEOUT`.

**Go `4dollama` (repo `cmd/4dollama`):** interactive `run` is **only** the plain Ollama-style `>>>` line REPL (bubble UI removed). Rebuild the binary after updates.
