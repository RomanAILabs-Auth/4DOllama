# Roma4D documentation hub

**Single-file edition:** **[Roma4D_Master_Guide.md](Roma4D_Master_Guide.md)** — all guides below merged with a full table of contents (install → ship → normative spec).

---

This folder is the **canonical home** for how-to guides: from **first install** to **shipping native products**, **LLM-assisted development**, and **debugging** when something breaks.

The **[Roma4D Elite Reference Guide](Roma4D_Guide.md)** remains the **normative spec** (what compiles, exact syntax, LLM hard rules). The guides here **route** you to the right section and add **workflow** context.

---

## Start here

| I want to… | Read |
|------------|------|
| **Everything in one document** | **[Roma4D_Master_Guide.md](Roma4D_Master_Guide.md)** |
| **Install** the toolchain (Windows / Linux / macOS) | **[Install_Guide.md](Install_Guide.md)** |
| **Understand** what the language is *for* (vision, applications) | **[User_Guide.md](User_Guide.md)** |
| **Write code** with correct syntax and rules | **[Roma4D_Guide.md](Roma4D_Guide.md)** |
| **See** all tools and versions (Go, Zig, clang, env vars) | **[Dependencies_Guide.md](Dependencies_Guide.md)** |
| **Use ChatGPT / Cursor** safely on `.r4d` | **[LLM_Guide.md](LLM_Guide.md)** |
| **Debug** failed builds and runs | **[Debugging_Guide.md](Debugging_Guide.md)** |
| **Decode** compiler / linker messages | **[Errors_Guide.md](Errors_Guide.md)** |
| **Ship** executables and releases | **[Shipping_Guide.md](Shipping_Guide.md)** |

---

## Guide map (elaborate track)

### 1. Install and environment

- **[Install_Guide.md](Install_Guide.md)** — Clone, prerequisites, one-command install, verify `r4d`, troubleshooting checklist, optional 4DOllama.
- **[Dependencies_Guide.md](Dependencies_Guide.md)** — Go, Zig, clang, MSYS2, `R4D_*` variables, `roma4d.toml`, optional Rust for 4DOllama.

### 2. Use the language

- **[User_Guide.md](User_Guide.md)** — Capabilities, “what can I build,” creative and technical directions (grounded vs roadmap).
- **[Roma4D_Guide.md](Roma4D_Guide.md)** — **Elite reference**: types, `par`, `spacetime:`, ownership, builtins, pipeline, **§20** examples index.

### 3. LLM-assisted development

- **[LLM_Guide.md](LLM_Guide.md)** — Which doc to paste, protocol, checklists, Expert vs strict.
- **Roma4D_Guide.md** — **§25–§29**: generation protocol, pre-submit checklist, **hard rules**, **Native AI Expert**.

### 4. Debugging and errors

- **[Debugging_Guide.md](Debugging_Guide.md)** — `R4D_DEBUG`, `last_build_failure.log`, `r4`/`r4d` flags, Expert terminal flow.
- **[Errors_Guide.md](Errors_Guide.md)** — How to read failures; pointer to **Roma4D_Guide §23** (error catalog).

### 5. Shipping products

- **[Shipping_Guide.md](Shipping_Guide.md)** — `r4 build`, artifacts, naming, bench, CI, licensing, optional container story for services.

---

## Monorepo pointers

| Topic | Location |
|--------|----------|
| Root **INSTALL** stub (quick link) | [../../INSTALL.md](../../INSTALL.md) |
| **4DOllama** product (HTTP API, `4dollama`) | [../../docs/4DOllama.md](../../docs/4DOllama.md) |
| Roma4D **README** (quick start, CLI) | [../README.md](../README.md) |

---

## Contributing to docs

- **Normative behavior** → change **Roma4D_Guide.md** and, if needed, compiler/tests.
- **Onboarding / tone / workflow** → change **User_Guide**, **Install_Guide**, or the hub **README** (this file).
- Keep **cross-links** between guides consistent when adding new pages.
- **Merged book:** after materially editing split guides, update **[Roma4D_Master_Guide.md](Roma4D_Master_Guide.md)** in the same change (or a follow-up) so the one-file edition stays aligned. There is no automatic merge step in CI yet.
