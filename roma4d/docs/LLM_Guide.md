# LLM Guide — working with Roma4D using AI assistants

Use this page to **paste the right document** into ChatGPT, Cursor, Copilot, etc., and to follow a **safe generation protocol** so the model emits **valid `.r4d`**, not “Python with 4D comments.”

---

## Golden rule

**Roma4D is not Python.** It is **ahead-of-time compiled** to **native code** via **LLVM**. There is **no** CPython, **no** f-strings, **no** `time()` wall clock, **no** arbitrary PyPI imports.

If an LLM drifts into Python, **stop** and re-paste **[Roma4D_Guide.md](Roma4D_Guide.md)** §**Mental model** and §**27 LLM hard rules**.

---

## What to attach / paste

| Goal | Primary document |
|------|------------------|
| **Generate or refactor `.r4d`** | **[Roma4D_Guide.md](Roma4D_Guide.md)** (full file if the tool allows — it is the spec). |
| **High-level “what is this language?”** | **[User_Guide.md](User_Guide.md)** + **Roma4D_Guide § Mental model**. |
| **Install / PATH / `r4d` not found** | **[Install_Guide.md](Install_Guide.md)**. |

Minimum for codegen: **Roma4D_Guide** sections **Mental model**, **Types**, **Builtins**, **`par`**, **`spacetime:`**, **Ownership 2.0**, **§22 Python vs Roma4D**, **§26 Pre-submit checklist**, **§27 Hard rules**.

---

## Step-by-step protocol (summary)

Full detail: **Roma4D_Guide §25** — *LLM code-generation protocol*.

1. Decide **entry shape**: `main() -> None` vs `-> int`.
2. Prefer **`list[vec4]` + `par for`** for parallel 4D simulation shapes unless the spec needs something smaller.
3. Use **only** APIs that appear in the Guide or in **`examples/`** / **`demos/`** indexed in **§20**.
4. **Strings:** static literals for `print` in portable programs; do not format numbers into strings in source.
5. **Time:** **`t`** / **`@ t`** are **compile-time staging**, not a runtime clock.
6. After drafting, run the **§26 pre-submit checklist** line by line.
7. Run **`r4d yourfile.r4d`** or **`r4 run --strict`** for CI-style raw errors.

---

## Native AI Expert (terminal)

When **`r4d`** / **`r4 run`** fails in **forgiving** mode, the toolchain can show a **structured debug block** (copy-paste friendly), optional hints, and a small **interactive** Q&A on a TTY.

**Spec:** **Roma4D_Guide §29** — *Native AI Expert*.

**CI / automation:** use **`r4 run --strict`** (or equivalent) so output stays **machine-parsable** without interactive prompts.

---

## Prompt snippets you can reuse

**System / preface:**

> You are generating **Roma4D** source (extension `.r4d`). It is **not** Python. Follow the attached **Roma4D Elite Reference Guide** exactly. If an API is not in the guide or indexed examples, do not invent it.

**After a bad generation:**

> Re-read **§27 LLM hard rules** and **§22 invalid patterns**. Rewrite the smallest program that compiles; use **`examples/min_main.r4d`** as a template.

---

## Related

- **Debugging:** [Debugging_Guide.md](Debugging_Guide.md)  
- **Errors:** [Errors_Guide.md](Errors_Guide.md)  
- **Hub:** [README.md](README.md)
