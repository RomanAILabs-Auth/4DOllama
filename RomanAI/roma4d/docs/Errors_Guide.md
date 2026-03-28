# Errors Guide — understanding Roma4D failures

This guide **orients** you; the **authoritative catalog** lives in the Elite Reference.

---

## Authoritative reference

**[Roma4D_Guide.md — §23 Compiler and linker error catalog](Roma4D_Guide.md)** (search for `## 23.` in-file)

Use that section when mapping a **specific** message to a **fix**.

---

## Failure surface (what you might see)

| Layer | Typical source |
|-------|----------------|
| **Lexer / parser** | Invalid syntax, wrong extension, BOM issues — see Guide §6–§8. |
| **Typecheck** | Wrong types, missing builtins, bad `par` capture — §7, §11, §13. |
| **Ownership 2.0** | SoA move/borrow violations — §11. |
| **MIR / LLVM** | Internal assert or bad lowering — often needs `last_build_failure.log` + issue upstream. |
| **Linker** | Missing **Zig** / **clang**, wrong **Windows** ABI, missing **`-lm`** (usually handled by driver on Unix). |

---

## Install-time vs compile-time

| Problem | Doc |
|---------|-----|
| **`go` / `r4d` not found** | [Install_Guide.md §4](Install_Guide.md#4-if-something-fails-checklist) |
| **`zig` / `clang` missing** | [Dependencies_Guide.md](Dependencies_Guide.md) |
| **Script execution policy (Windows)** | [Install_Guide.md](Install_Guide.md) |

---

## Workflow

1. Capture **full** output with **`r4 run --strict`** (or save terminal log).  
2. Open **`roma4d/debug/last_build_failure.log`** if present.  
3. Match tokens to **Roma4D_Guide §23**.  
4. If stuck, paste **§23** + your log into an LLM **with** [LLM_Guide.md](LLM_Guide.md) constraints.

**Debugging workflow:** [Debugging_Guide.md](Debugging_Guide.md)
