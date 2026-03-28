# RomanAI — agent handoff (for Gemini / external assistants)

**Purpose:** Single file to paste or attach so another model knows repo state, constraints, and how to run things without rediscovering prior debugging.

---

## What this project is

- **RomanAI:** A Roma4D (**R4D / `.r4d`**)–based CLI around a **GGUF** model path. Marketing framing: “4D-native” runner; implementation mixes compiled R4D with a **C runtime** in `roma4d/rt/` (GGUF layout, lattice hooks, vocab print).
- **Strict product rules (from product owner):** Offline local CLI only — no server, HTTP, Ollama, or **llama.cpp** in the RomanAI narrative. Kernel logic lives in **pure `.r4d`** sources; the **binary name must be `romanai`** when built via the RomanAI Makefile flow.
- **Toolchain:** **Go** builds `r4`, `r4d`, and optionally **`romanai`** from `RomanAI/roma4d`. **Zig/clang** may be required for linking (see Roma4D guide / `debug/last_build_failure.log`).

---

## Repository layout (what matters)

| Path | Role |
|------|------|
| `RomanAI/src/cli/main.r4d` | **Canonical kernel source** — edit here first. |
| `RomanAI/r4d/romanai_main.r4d` | **What `r4d run` / launchers execute** — synced from `main.r4d` + header. |
| `RomanAI/romanai.r4d` | Copy of `main.r4d` for `make build` / `r4 build romanai.r4d`. |
| `RomanAI/roma4d/` | **Authoritative Roma4D compiler + RT** for RomanAI work (same Go module as root `4DEngine/roma4d` in many setups — keep them aligned if both exist). |
| `RomanAI/scripts/romanai.ps1` | Launcher: `go run ./cmd/r4d run <kernel>` from `roma4d/` (avoids stale global `r4d`). |
| `RomanAI/romanai.cmd` | Calls `scripts/romanai.ps1` — run **from `RomanAI/`**, not from `roma4d/`. |
| `RomanAI/roma4d/cmd/romanai/main.go` | **`go install ./cmd/romanai`** — finds checkout via `ROMANAI_ROOT` or walking cwd; same `go run` behavior as the ps1. |
| `RomanAI/scripts/Sync-RomanAIKernel.ps1` | Copies `src/cli/main.r4d` → `romanai.r4d` and `r4d/romanai_main.r4d`. **Run after editing `main.r4d`.** |
| `RomanAI/scripts/Install-RomanAI-R4.ps1` | `go install ./cmd/r4 ./cmd/r4d ./cmd/romanai`. |
| `RomanAI/Makefile` | `build` / `run` / `test-final` / `clean` (uses `r4` when on PATH). |
| `RomanAI/src/lattice/coupling.r4d`, `src/loader/gguf.r4d`, `src/engine/core.r4d` | Reference / shard sources; **host kernel is single-unit** (no cross-file imports in the synced kernel today). |
| `RomanAI/roma4d/rt/romanai_gguf_layout.c` (and twin under `4DEngine/roma4d`) | Emits **`ROMANAI_LATTICE`** / **`ROMANAI_DECODE_GRAPH`** lines to **stderr** in some paths (e.g. `layout_not_loaded`). |

---

## Current behavioral state (as of last handoff)

1. **Stale `r4d` on PATH** caused `NameError: mir_romanai_cli_model_path` / `mir_romanai_vocab_print` — those builtins exist only in the **RomanAI-aligned** `roma4d` tree. Fix: use **`RomanAI/romanai.cmd`**, or **`go install`** from **`RomanAI/roma4d`**, or the **`romanai`** Go wrapper + ensure `%GOPATH%\bin` wins on PATH.
2. **`.gguf` not `.ggff`** — extension check in launcher.
3. **Env:** Launchers set **`ROMANAI_CLI_MODEL`**, **`ROMANAI_GGUF`**, **`ROMANAI_PROMPT`** for the RT.
4. **Prompt handling:** `sub == "run"` uses placeholder prompt; **else** `prompt = sub` (avoid fragile `str != ""` comparisons in this pipeline).
5. **Kernel (`src/cli/main.r4d`)** still calls **`mir_romanai_lattice_coupling_step`** and **`mir_romanai_decode_graph_step`** each decode step — when GGUF layout is not fully loaded, stderr can spam **lattice/decode “error”** lines. **Phase 10 (silent lattice / remove those messages)** was specified by the product owner but **may not yet be merged**; confirm in `main.r4d` and in `gguf.r4d` (`dequantize_to_4d`).
6. **`loader/gguf.r4d`** still referenced decode/lattice MIR calls in at least one path — same stderr risk if that code path runs.

---

## Commands cheat sheet (Windows)

```powershell
# From repo RomanAI folder (recommended)
cd C:\...\4DEngine\RomanAI
.\romanai.cmd run C:\path\to\model.gguf "optional prompt"

# After editing src/cli/main.r4d
.\scripts\Sync-RomanAIKernel.ps1

# Install tools into Go bin
cd .\roma4d
go install ./cmd/r4 ./cmd/r4d ./cmd/romanai

# From home directory without cd into repo
$env:ROMANAI_ROOT = 'C:\...\4DEngine\RomanAI'
romanai run C:\path\to\model.gguf "prompt"
```

---

## What a new agent should do first

1. Read **`src/cli/main.r4d`** and **`r4d/romanai_main.r4d`** — confirm they match (or run sync script).
2. Grep for **`mir_romanai_`** to see all host builtins the kernel depends on.
3. If cleaning stderr: grep **`ROMANAI_LATTICE`** / **`ROMANAI_DECODE_GRAPH`** in **`romanai_gguf_layout.c`** and trace call sites from R4D.
4. Follow **`docs/Roma4D_Guide.md`** (or `RomanAI/roma4d/docs/Roma4D_Guide.md`) for §22 / §26 / §27 constraints when editing `.r4d`.

---

## Open / follow-up (optional)

- Phase 10: silent professional output (no lattice/decode error lines); possibly pure-R4d gravity path without those MIR steps when layout missing.
- Ensure **`4DEngine/roma4d`** and **`RomanAI/roma4d`** stay in sync if both are used for `go install`.

---

*Generated for handoff between human + coding agents; update this file when milestones land.*
