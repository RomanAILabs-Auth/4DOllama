# RomanAI
The first true 4D-native GGUF runner built purely in R4D.

`romanai run filename.gguf` — streams word-for-word with Cl(4,0) spacetime intelligence.

## Quickstart

Build the self-hosted binary (same sources as `src/cli/main.r4d`; root `romanai.r4d` is kept in sync):

```bash
make build
```

The R4D runtime reads **`ROMANAI_GGUF`** and **`ROMANAI_PROMPT`** (see `roma4d/rt/roma4d_rt.c`). Examples:

```bash
# Version
ROMANAI_PROMPT=version ./romanai

# List models (placeholder)
ROMANAI_PROMPT=list ./romanai

# Chat-style batch line (set model path + message)
ROMANAI_GGUF=models/your.gguf ROMANAI_PROMPT="Hello" ./romanai

# Run-style prompt text
ROMANAI_PROMPT=run ./romanai
```

```bash
romanai run filename.gguf
romanai chat
```

**Windows (recommended):** from the `RomanAI` folder use the bundled toolchain so you never pick up an old global `r4d`:

```powershell
.\romanai.cmd run C:\path\to\model.gguf
.\romanai.cmd run C:\path\to\model.gguf Your prompt words here
```

Optional: refresh `r4d` / `r4` on your PATH from this repo: `.\scripts\Install-RomanAI-R4.ps1`

For `romanai run` / `romanai chat` as above, use a small shell wrapper that sets `ROMANAI_GGUF` / `ROMANAI_PROMPT` before invoking the compiled binary.

### Host launcher (`romanai run model.gguf`)

If your PATH `romanai` reports **kernel missing** for `RomanAI\r4d\romanai_main.r4d`, that file must exist. It is generated from `romanai.r4d`:

```powershell
cd C:\Users\Asus\Desktop\4DEngine\RomanAI
.\scripts\Sync-RomanAIKernel.ps1
```

After editing `romanai.r4d`, run the sync script again so the kernel stays up to date.

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) in this directory (same text as the monorepo root and `roma4d/LICENSE`).

## Contact

**RomanAILabs** — *Daniel Harding*  
[romanailabs@gmail.com](mailto:romanailabs@gmail.com) · [daniel@romanailabs.com](mailto:daniel@romanailabs.com)
