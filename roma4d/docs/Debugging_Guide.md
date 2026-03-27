# Debugging Guide — Roma4D builds and runs

When **`r4d`**, **`r4 run`**, or **`r4 build`** fails, use this flow before opening an issue or asking an LLM.

---

## 1. Reproduce with clear flags

| Command | When to use |
|---------|-------------|
| **`r4d file.r4d`** | Default: forgiving mode may invoke **Native AI Expert** hints on failure. |
| **`r4 run --strict file.r4d`** | **CI / logs** — raw compiler errors, no interactive fluff. |
| **`r4 run -bench file.r4d`** | See **per-phase timings** (lexer → link → **native_run**). |
| **`R4D_DEBUG=1`** (Unix) or set user env on Windows | Extra stderr diagnostics where implemented. |

---

## 2. Read the build failure log

On many failures the driver writes:

**`roma4d/debug/last_build_failure.log`**

It typically captures:

- Linker command line (**`zig cc`** or **`clang`**)  
- **stderr** from the linker  
- A **head** of generated **LLVM IR** (useful for codegen bugs)

Open this file **first** when the terminal output scrolls away.

---

## 3. Common classes of failure

| Symptom | Check |
|---------|--------|
| **Linker not found** | **Zig** or **clang** on `PATH`; **`R4D_ZIG`**; **`R4D_GNU_ROOT`** (Windows clang path). |
| **`roma4d.toml` not found** | Run from a tree that contains **`roma4d.toml`** upward from the `.r4d` path, or set **`R4D_PKG_ROOT`**. |
| **Type / ownership errors** | **Roma4D_Guide §11** Ownership 2.0; **§10** SoA / aos. |
| **Illegal Python-ism** | **Roma4D_Guide §22** — f-strings, dynamic imports, `while` growth patterns, etc. |

---

## 4. Native AI Expert (interactive)

In default **forgiving** mode, the CLI may print a **rich debug block** and offer **patch hints** or Q&A.

**Full behavior:** **Roma4D_Guide §29**.

For **scripts and CI**, always prefer **`--strict`**.

---

## 5. LLVM / MIR introspection

Advanced:

- Inspect **MIR** and **LLVM IR** expectations in **`roma4d/tests/`** and compiler sources.  
- **`-bench`** helps separate **compile** vs **run** time.

---

## 6. Related docs

- **Error message catalog:** **Roma4D_Guide §23** — also summarized in [Errors_Guide.md](Errors_Guide.md).  
- **Install / PATH:** [Install_Guide.md](Install_Guide.md).  
- **LLM-assisted fix:** [LLM_Guide.md](LLM_Guide.md).
