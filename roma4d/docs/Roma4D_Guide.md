# Roma4D Elite Reference Guide (Human + LLM)

**Audience:** Expert practitioners and **automated code generators** (LLMs). Treat this document as a **normative specification** for what **compiles and links today** in toolchain **0.1.0**. If something is not described here as supported, **do not emit it**—prefer a smaller, boring program that builds.

**How-to hub (install, use, LLM workflow, debugging, shipping):** [README.md](README.md) · **All-in-one book:** [Roma4D_Master_Guide.md](Roma4D_Master_Guide.md)

**Official source extension:** **`.r4d`**. Legacy **`.r4s`** and **`.roma4d`** are still accepted by the driver. **`.py` is not Roma4D.**

**Edition:** `2025` in `roma4d.toml` names the staged feature set.

**Not Python:** Roma4D uses **Python 3.12–shaped surface syntax** (indentation, `def`, `class`, many keywords) but is a **separate, ahead-of-time compiled** language targeting **LLVM IR → native executable**. There is **no** CPython interpreter, **no** arbitrary `import` of PyPI packages, **no** f-strings, **no** `time()` wall clock, **no** dynamic `eval`.

---

## How an LLM should use this document

1. **Read [Mental model](#mental-model-read-this-first) and [LLM hard rules](#27-llm-hard-rules-non-negotiable) first.** If the user asks for “Python with 4D,” **refuse** and output valid **`.r4d`** per this spec.
2. **Before writing code**, decide: entry type `main() -> None` vs `-> int`, whether imports are needed (`from libgeo import ...`), and whether the program uses **`list[vec4]` + `par for`** (safe default for “4D simulation” asks).
3. **After drafting**, run the **[pre-submit checklist](#26-pre-submit-checklist-copy-for-every-generation)** mentally line-by-line.
4. **On uncertainty**, choose the **smallest** construct that appears in **`examples/hello_4d.r4d`** or **`demos/spacetime_collider.r4d`**. Never invent APIs.
5. **Strings:** only **static** string literals for `print` in portable programs. Do not format numbers into strings in Roma4D source.
6. **Time:** **`t`** and **`expr @ t`** are **compile-time staging** tokens—not a runtime clock.

---

## Table of contents

- [Mental model (read this first)](#mental-model-read-this-first)

**Core toolchain**

1. [How the compiler runs](#1-how-the-compiler-runs)
2. [Project layout and `roma4d.toml`](#2-project-layout-and-roma4dtoml)
3. [CLI commands, PATH, and package root](#3-cli-commands-path-and-package-root)
4. [Modules and imports](#4-modules-and-imports)

**Language surface**

5. [Entry point, functions, and control flow](#5-entry-point-functions-and-control-flow)
6. [Lexical structure: tokens, keywords, literals](#6-lexical-structure-tokens-keywords-literals)
7. [Types (complete table)](#7-types-complete-table)
8. [Builtins and constructors (authoritative)](#8-builtins-and-constructors-authoritative)
9. [Geometric algebra (Cl(4,0)) operators](#9-geometric-algebra-cl40-operators)
10. [Classes, `soa`, and `aos`](#10-classes-soa-and-aos)
11. [Ownership 2.0 (linear SoA, borrows, taint)](#11-ownership-20-linear-soa-borrows-taint)
12. [Spacetime: `t`, `@ t`, `spacetime:`](#12-spacetime-t--t-spacetime)
13. [Parallelism: `par for`](#13-parallelism-par-for)
14. [Lists and comprehensions](#14-lists-and-comprehensions)
15. [Systems: `unsafe:` and MIR hooks](#15-systems-unsafe-and-mir-hooks)

**Runtime and platform**

16. [Native runtime (`rt/roma4d_rt.c`)](#16-native-runtime-rtroma4d_rtc)
17. [Ollama / HTTP demos (builtins)](#17-ollama--http-demos-builtins)
18. [Compilation pipeline](#18-compilation-pipeline)
19. [Debugging, logs, and common failures](#19-debugging-logs-and-common-failures)

**LLM operations manual**

20. [Example programs (indexed)](#20-example-programs-indexed)
21. [File extensions](#21-file-extensions)
22. [Python vs Roma4D — invalid pattern encyclopedia](#22-python-vs-roma4d--invalid-pattern-encyclopedia)
23. [Compiler and linker error catalog](#23-compiler-and-linker-error-catalog)
24. [Ergonomics: `r4`, `GOBIN`, `R4D_PKG_ROOT`, embedded root](#24-ergonomics-r4-gobin-r4d_pkg_root-embedded-root)
25. [LLM code-generation protocol (step-by-step)](#25-llm-code-generation-protocol-step-by-step)
26. [Pre-submit checklist (copy for every generation)](#26-pre-submit-checklist-copy-for-every-generation)
27. [LLM hard rules (non-negotiable)](#27-llm-hard-rules-non-negotiable)
28. [Document maintenance](#document-maintenance)
29. [Native AI Expert (terminal debug + interactive)](#29-native-ai-expert-terminal-debug--interactive)
30. [Spacetime engine foundations (theory manuscript)](#30-spacetime-engine-foundations-theory-manuscript)

---

## Mental model (read this first)

Roma4D is built around three core ideas:

1. **Columns, not heap objects**  
   Hot data is expressed as **SoA fields** and **`list[vec4]` worldtubes** scanned by **`par for`**, not pointer graphs.

2. **Time is explicit in the language**  
   **`t`**, **`expr @ t`**, and **`spacetime:`** blocks express *when* (compile-time narrative / MIR staging). They **do not** add a hidden interpreter loop in the emitted binary for ordinary programs.

3. **Parallelism is first-class**  
   **`par for`** marks structured parallel loops with ownership-aware capture rules and backend hints (SIMD / future GPU paths).

If you model problems as **vectors + rotors + parallel updates**, you stay on the supported path.

---

## 1. How the compiler runs

**Inputs**

- One **primary** `.r4d` / `.r4s` / `.roma4d` file per `r4 build` or `r4 run`.
- Additional files reached via **`from foo import ...`** under the **package root** (directory containing **`roma4d.toml`**).

**Pipeline (conceptual)**

`lexer → parser → typecheck → Ownership 2.0 → MIR → LLVM IR → linker`

**Linker (platform)**

- **Windows (default):** **`zig cc`** when `zig` / `zig.exe` is on `PATH`. Override with **`R4D_ZIG`** pointing at the Zig executable.
- **Windows (fallback):** **LLVM `clang`** targeting **`*-windows-gnu`** (MinGW ABI), plus MinGW headers—see §19.
- **Unix:** **`clang`**; math requires **`-lm`** when linking `roma4d_rt.c` (driver handles this).

**Output:** Native executable (`.exe` on Windows).

**Backend:** `[build] backend = "llvm"` in `roma4d.toml`. CUDA / full GPU execution is roadmap; MIR may carry GPU-oriented metadata.

---

## 2. Project layout and `roma4d.toml`

Typical layout (this repository):

```text
roma4d/
  roma4d.toml          # required manifest at package root
  r4d.ps1              # Windows: build + run r4d (or use installed r4d on PATH)
  r4d.cmd              # cmd.exe wrapper for r4d.ps1
  cmd/r4d/             # CLI entry
  internal/cli/        # shared Main() for r4 / r4d / roma4d
  src/compiler/        # typecheck, ownership, MIR, LLVM, link
  src/parser/          # lexer + parser
  rt/roma4d_rt.c       # C runtime linked into user programs
  libgeo.r4d           # example importable module
  examples/*.r4d
  demos/*.r4d
  debug/               # last_build_failure.log (generated on failure)
```

**`roma4d.toml` fields (used today)**

| Section | Field | Role |
|--------|--------|------|
| top | `name`, `version`, `edition` | Package identity; `edition` labels language stage. |
| `[package]` | `authors`, `description` | Metadata. |
| `[build]` | `default_backend`, `backend`, `incremental` | LLVM backend selection. |
| `[systems]` | `gc = false`, `unsafe = true` | No GC; `unsafe:` blocks allowed at language level. |

---

## 3. CLI commands, PATH, and package root

### Simplest usage (remember this)

After **one-time** Windows setup (see below), you only run:

```text
r4d myfile.r4d
r4d C:\full\path\to\myfile.r4d
```

That is the same as **`r4d run …`**. Use **quotes** around the path if the folder name has spaces.

**One-time Windows setup:** from the **`roma4d`** repo folder in PowerShell:

```powershell
.\scripts\Install-R4dUserEnvironment.ps1
```

Then **close that window** and open a **new** PowerShell so **`PATH`** updates. The script installs **`r4`**, **`r4d`**, **`roma4d`** into **`go env GOBIN`** (e.g. conda **`Scripts`**) or **`GOPATH\bin`**, sets **`R4D_PKG_ROOT`**, and embeds the repo path in **`r4d.exe`** so **`r4d`** works even when your **`.r4d`** file is not inside the clone.

**From the repo without reinstalling:** **`r4d.ps1`** (PowerShell) or **`r4d.cmd`** (cmd.exe) in the **`roma4d`** directory rebuilds the tools into your Go bin and forwards arguments to **`r4d.exe`**. Run with **no arguments** to see a short “how to run” reminder plus **`r4d help`**.

**Binaries:** **`r4`**, **`r4d`**, **`roma4d`** share one implementation (`internal/cli`). Behavior is identical aside from the banner label. **`r4d help`** starts with the same “simplest way” lines for humans.

| Invocation | Meaning |
|------------|---------|
| `r4 help` | Usage text (stderr). |
| `r4 version` | `roma4d (r4d) 0.1.0 <os>/<arch>`. |
| `r4d sketch.r4d` [args…] | **Shorthand for `r4d run`** (same Expert / `--strict` rules). |
| `r4 run [--strict] file.r4d [-bench] [args…]` | Temp dir build, run executable. |
| `r4 build [--strict] file.r4d [-o out] [-bench]` | Emit persistent executable. |

**`--strict`:** Disables the **Native AI Expert** (no rich debug block, no interactive session); raw compiler/linker errors only. Use in CI. **`--forgiving`** (default) re-enables Expert after **`--strict`** if both appear — **later flag wins**.

**`-bench`:** Prints phase timings (`load_manifest`, `parse`, `typecheck`, LLVM, `zig_*` or `clang_*`, and `native_run` for `run`).

### Package root resolution (`findPkgRoot`)

Order of resolution:

1. Walk **upward** from the **source file’s directory** until **`roma4d.toml`** is found.
2. If not found, try **`R4D_PKG_ROOT`**, then **`ROMA4D_HOME`** (must contain `roma4d.toml`).
3. If still not found, use **`EmbeddedPkgRoot`** (set at **link time** when built via `install.ps1`, `install.sh`, or `scripts/Install-R4dUserEnvironment.ps1` with `-ldflags -X .../cli.EmbeddedPkgRoot=...`).

This is how **`r4d C:\Desktop\foo.r4d`** works **without** `cd` into the clone after a proper install.

### Relative paths and cwd

- **`r4 run demos\foo.r4d`** resolves **`demos\foo.r4d`** relative to the **shell current working directory**, not relative to `R4D_PKG_ROOT`.
- **Imports** (`from libgeo import ...`) resolve under **package root**, not next to the sketch file.

### Windows PowerShell

Do **not** paste the **`PS C:\...>`** prompt into the terminal; on some systems **`PS`** is an alias and breaks the line.

### Expert mode (forgiving, default)

On **`.r4d` build/run failure**, the CLI invokes the **Native AI Expert** (`src/ai/expert.go`): a **copy-pasteable terminal debug block**, **`debug/last_build_failure.log`** append, optional **patch suggestion + [y/N] prompt**, and an **interactive** Q&A when **stdin is a TTY**. Full behavior is specified in **§29**.

---

## 4. Modules and imports

### 4.1 Module file resolution

For `import foo.bar` or `from foo.bar import x`, the driver tries paths under **package root** (see `ResolveRoma4DModuleFile` in `src/compiler/source_ext.go`):

**Per segment / dotted name, extension order is always:** **`.r4d` → `.r4s` → `.roma4d`**.

Examples (conceptual):

- `pkgRoot/foo/bar.r4d` (or `.r4s` / `.roma4d`)
- `pkgRoot/foo.bar.r4d` (dotted flat file)

Missing file → **`ImportError`**.

### 4.2 Import forms

| Form | Supported |
|------|-----------|
| `import mod` | Yes (with optional `as`). |
| `from mod import a, b, c` | Yes. |
| `from mod import *` | **No** — hard error in typechecker. |

### 4.3 Exports (v0 behavior)

Importable names are **top-level** **`def`** and **`class`** in the module file. Prefer defining public APIs at module scope.

### 4.4 Example library (`libgeo.r4d`)

```roma4d
def bump(n: int) -> int:
    return n + 1

def identity_v4(v: vec4) -> vec4:
    return v
```

Consumer:

```roma4d
from libgeo import bump, identity_v4
```

---

## 5. Entry point, functions, and control flow

### 5.1 `main`

- Entry point is **`def main()`** (no `if __name__ == "__main__"`).
- **`def main() -> int:`** must **`return`** an integer on all paths (otherwise **synthesized return** warning).
- **`def main() -> None:`** may use **`return None`**, bare **`return`**, or fall off the end; lowering targets **void** `main` in LLVM.

### 5.2 Function definitions

- Use **`def name(args) -> ReturnType:`** with **annotated** parameters and return for public / non-trivial functions.
- **Variadic** Python features beyond what examples use: **avoid** unless you have verified with `r4d --strict`.

### 5.3 Statements (supported idioms)

Use **indentation-based** blocks (spaces; keep consistent—**do not mix tabs**).

| Construct | Roma4D use |
|-----------|------------|
| `if` / `elif` / `else` | Yes. |
| `while` | Yes. |
| `for x in iterable:` | Yes; **`range(...)`** is the common iterable. |
| `break` / `continue` | Parsed; use like Python when needed. |
| `pass` | Yes (including inside **`spacetime:`**). |
| `return` | Yes. |
| `try` / `except` / `finally` | Lexed; **do not rely** for LLM-generated code—prefer compile-time clean code. |
| `async` / `await` | Lexed; **not** a supported async runtime model—**do not generate**. |
| `lambda` | Parsed; typechecking is **not** a first-class Python lambda story—**do not generate**; use **`def`**. |
| `match` / `case` | Lexed; **avoid** unless you verify. |

---

## 6. Lexical structure: tokens, keywords, literals

### 6.1 Comments and docstrings

- **`#` line comments** anywhere.
- **Triple-quoted strings** used as **class docstrings** appear in sources (e.g. `hello_4d.r4d`); treat as documentation only.

### 6.2 Identifiers

Standard Python-like identifiers.

### 6.3 Numeric literals

- **Integers** and **floats** as in Python.
- **Underscore separators** allowed: **`1_000_000`** (see demos).

### 6.4 String literals

- **`"..."` and `'...'`** only.
- **No f-strings** (`f"..."`) — **not** the string model.
- **No** automatic formatting of **`int` / `float`** for human display in portable Roma4D—use **fixed messages** (`print("tick 512")`).

### 6.5 Keywords (representative set)

Includes Python keywords **plus** Roma4D extensions:

**Roma4D-specific (non-negotiable spelling):** `par`, `soa`, `aos`, `vec4`, `rotor`, `multivector`, `borrow`, `mutborrow`, `unsafe`, **`t`** (time coordinate), **`spacetime`**.

**Also reserved / lexed as keywords:** `def`, `class`, `return`, `if`, `else`, `elif`, `for`, `while`, `pass`, `break`, `continue`, `import`, `from`, `as`, `True`, `False`, `None`, `and`, `or`, `not`, `in`, `is`, `try`, `except`, `raise`, `with`, `async`, `await`, `lambda`, `match`, `case`, …

**LLM rule:** If unsure whether a Python keyword works end-to-end, **do not use it**—stick to constructs in **`examples/hello_4d.r4d`**.

---

## 7. Types (complete table)

### 7.1 Host / scalar types

| Type | Notes |
|------|------|
| `int` | Signed integer. |
| `float` | Double-precision float in native lowering. |
| `str` | String; **`print`** is oriented toward **string** arguments in the Roma4D runtime. |
| `bool` | `True` / `False`. |
| `none` | Type of **`None`**; also used where no value is returned. |

### 7.2 4D / systems types

| Type | Role |
|------|------|
| `vec4` | 4-component vector; geometric ops with `rotor` / `multivector`. |
| `rotor` | Rotor value; construct with **`rotor(angle=..., plane="...")`**. |
| `multivector` | General multivector; **`multivector()`** default ctor in examples. |
| `rawptr` | Raw pointer for `unsafe` + **`mir_*`**. |
| `time` | Type of **`t`** and time-coordinate reasoning. |

### 7.3 Constructors / generics in surface syntax

- **`list[T]`** — e.g. **`list[vec4]`**, **`list[int]`**.
- **Do not** use C/Rust array syntax like **`[T; N]`** for bulk data.

---

## 8. Builtins and constructors (authoritative)

Defined in **`seedBuiltins`** (`src/compiler/typechecker.go`). Names are **case-sensitive**.

| Name | Callable shape | Result (declared) | Notes |
|------|----------------|-------------------|-------|
| `print` | variadic | `none` | Runtime **`puts`** style—stick to **string literals** for portable code. |
| `range` | variadic | `list[int]`-like iterable | Used in **`for i in range(n):`** and comprehensions. |
| `len` | `(any)` | `int` | |
| `int`, `float`, `str`, `bool` | variadic ctor/cast | respective scalar type | |
| `abs` | `(any)` | `any` (loose) | Prefer for numerics only when needed. |
| `vec4` | keyword args | `vec4` | e.g. **`vec4(x=0, y=0, z=0, w=1)`** or **`w=1.0`**. |
| `rotor` | keyword args | `rotor` | **`rotor(angle=float, plane=str)`** — demos use **`"xy"`**, **`"yz"`**, **`"xw"`**. |
| `multivector` | variadic | `multivector` | Often **`multivector()`** empty. |
| `borrow` | `(x)` | `any` | **Argument must be a simple name** for borrow bookkeeping. |
| `mutborrow` | `(x)` | `any` | Same restriction pattern. |
| `timetravel_borrow` | `(x)` | `any` | Used in **`spacetime:`** regions in demos. |
| `mir_alloc` | `(size: int)` | `rawptr` | |
| `mir_ptr_store` | `(ptr: rawptr, value: int)` | `none` | |
| `mir_ptr_load` | `(ptr: rawptr)` | `int` | |
| `ollama_demo` | `()` | `int` | See §17. |
| `quantum_server_demo` | `()` | `int` | See §17. |
| `True`, `False`, `None` | literals | `bool` / `none` | |

**There is no** `time()`, **`datetime`**, **`random`**, **`open()`**, **`requests`**, **`numpy`**, **`torch`**.

---

## 9. Geometric algebra (Cl(4,0)) operators

On **`vec4`**, **`rotor`**, **`multivector`** (where implemented), these operators are **geometric**:

| Operator | On 4D types | On plain `int` |
|----------|-------------|----------------|
| `*` | Geometric product (e.g. **`vec4 * rotor`**) | Integer multiply |
| `^` | Outer product | **XOR** |
| `|` | Contraction / inner-style product | **OR** |

**Type-directed disambiguation** chooses algebra vs bitwise. **Do not** mix metaphors inside one expression without matching types.

**Pattern from `examples/hello_4d.r4d`:**

```roma4d
a: vec4 = pos[0] * rot
b: multivector = pos[0] ^ demo_mv
c: float = pos[0] | demo_mv
i: int = 3 ^ 5
j: int = 1 | 2
```

---

## 10. Classes, `soa`, and `aos`

```roma4d
class Particle:
    soa pos: vec4
    soa vel: vec4
```

- **`soa`** declares **column / linear** fields with Ownership 2.0 semantics (§11).
- **`aos`** exists as a layout keyword; **prefer `soa`** unless you have a verified pattern.

**Construction:** **`Particle()`** lowers to a runtime symbol (see `roma4d_rt.c`).

---

## 11. Ownership 2.0 (linear SoA, borrows, taint)

**All enforcement is compile-time.** The native binary does not execute a Rust-style borrow checker at runtime.

### 11.1 Linear `soa` field access (canonical pattern)

```roma4d
cell: Particle = Particle()
col: vec4 = cell.pos       # move out
col = col * rot
cell.pos = vec4(x=0, y=0, z=0, w=1)   # write back before next read
again: vec4 = cell.pos
```

**Failure modes**

| Diagnostic | Meaning | Fix |
|------------|---------|-----|
| `UseAfterMoveError` | Read **`soa`** field again without re-assigning. | Assign **`cell.pos = ...`** before second read. |
| `BorrowError` | Overlapping **`borrow` / `mutborrow`**. | Narrow borrow scope; never borrow the same name conflictingly. |
| `TaintError` | Value flowed through **`print(...)`** then assigned into **linear / `soa`** slot. | Do not connect `print` outputs to **`soa`** writes. |

### 11.2 `borrow`, `mutborrow`, `timetravel_borrow`

- Operand must be a **simple identifier** matching the borrow pass expectations.
- **`timetravel_borrow(rotor)`** appears inside **`spacetime:`** in **`hello_4d.r4d`** as a chronology narrative hook.

---

## 12. Spacetime: `t`, `@ t`, `spacetime:`

**Compile-time staging:** these constructs participate in **static** MIR / type reasoning. They **do not** magically run a 4096-tick interpreter for you unless you wrote explicit loops yourself in normal statement code.

**`t` coordinate**

```roma4d
_tau: time = t
sample: vec4 = worldtube[0] @ t
```

**`spacetime:` regions**

```roma4d
spacetime:
    par for p in worldtube:
        p = p * rot
    _hold = timetravel_borrow(rot)
```

**Multiple `spacetime:` blocks** at the **same** indentation level are used in **`demos/spacetime_collider.r4d`** (“frames”). That is **sequential** regions, not nesting.

**LLM rules**

- **Do not** place **`spacetime:`** **inside** a runtime **`for`** loop unless you **know** it is valid for your target—**not** demonstrated in canonical demos.
- **Do not** expect **`spacetime:`** alone to iterate **tick** counters—use normal **`for tick in range(n):`** **outside** if you need counted steps, and use **`print("literal")`** for each milestone if you cannot format integers.

---

## 13. Parallelism: `par for`

```roma4d
spacetime:
    par for p in positions:
        p = p * rot
```

**Semantics (today)**

- Structured parallel loop with **sendability** / capture checking in the ownership pass.
- SIMD-friendly lowering exists for some **`vec4` * `rotor`** patterns.
- Full CUDA GPU execution is **not** the default end-to-end experience.

**Critical pattern:** iterate **directly** over **`list[vec4]`** binding **`p`** as the element, then assign **`p = p * rot`**. **Do not** rewrite as **`par for i in range(N): cosmos[i] = ...`** unless you **know** it is supported—the happy path is the **`par for p in cosmos`** form (see §22).

---

## 14. Lists and comprehensions

**Construction**

```roma4d
xs: list[vec4] = [vec4(x=0, y=0, z=0, w=1) for _ in range(N)]
```

**Indexing**

- **`xs[i]`** reads an element.
- Updates inside **`par for p in xs`** should use the **`p = ...`** pattern, not index loops, unless you **verify**.

**LLM rule:** **`list[vec4]`** scales with memory—**1_000_000** is the canonical demo scale; **10_000_000** appears in **`demos/cosmic_genesis.r4d`** and is heavier.

---

## 15. Systems: `unsafe:` and MIR hooks

```roma4d
unsafe:
    rawp: rawptr = mir_alloc(128)
    mir_ptr_store(rawp, 42)
    _peek: int = mir_ptr_load(rawp)
```

Requires **`[systems] unsafe = true`**. **`gc = false`**—no tracing GC.

---

## 16. Native runtime (`rt/roma4d_rt.c`)

Linked from **`pkgRoot/rt/roma4d_rt.c`**. Provides C symbols used by LLVM lowering, including **`print` → `puts`**, **`vec4`**, **`rotor`**, **`multivector`**, **`Particle`**, and demo hooks calling **`curl`** for Ollama.

**Numeric printing:** not a general **`printf` bridge** from Roma4D for arbitrary **`float`** formatting—use **fixed strings** or extend C if you need digits.

---

## 17. Ollama / HTTP demos (builtins)

### `ollama_demo()`

- Requires **`curl`** on `PATH` and local **Ollama** (`ollama serve`, default `http://127.0.0.1:11434`).
- JSON body is **fixed in C** (dynamic string building from `.r4d` is not the general story).

### `quantum_server_demo()`

- Same infrastructure.
- Env: **`QUANTUM_QUERY`**, **`QUANTUM_CONTINUE=1`** (see `demos/quantum_server.r4d`).

**Security:** uses **`system("curl ...")`** from C—trusted local dev only.

---

## 18. Compilation pipeline

1. Parse primary + imported modules.
2. Typecheck + resolve imports + attach builtins.
3. Ownership 2.0 pass.
4. Lower AST → MIR.
5. MIR → LLVM IR.
6. **`zig cc` or `clang`** compile `.ll` → `.o`, link **`roma4d_rt.c`** → executable.

---

## 19. Debugging, logs, and common failures

### `debug/last_build_failure.log`

On failure, a structured record is appended under **package root** **`debug/last_build_failure.log`**: stage, argv, stderr, LLVM IR head. In **forgiving** mode, the **Native AI Expert** also appends a block under stage **`ai_expert`** (see **§29**).

### `R4D_DEBUG=1`

Mirrors failure details to stderr immediately (Unix / Windows / PowerShell).

### Zig vs Clang (Windows)

- **Install Zig** and put it on **`PATH`**—simplest path.
- **Clang fallback:** if you see **`mm_malloc.h`** errors, set **`R4D_GNU_ROOT`** to your MinGW prefix (e.g. `C:\msys64\ucrt64`) or **install Zig**.

### Linux `sqrt` undefined

Preserve **`-lm`** when customizing links; the driver adds it on non-Windows.

### Indentation errors

Mixed tabs/spaces cause **`INDENT` / `DEDENT`** parse failures—reformat uniformly.

---

## 20. Example programs (indexed)

| Path | What it proves |
|------|----------------|
| `examples/min_main.r4d` | Smallest **`main() -> int`**, exit **42**. |
| `examples/hello_4d.r4d` | **Imports**, **GA ops**, **SoA**, **`t` / `@ t`**, **`spacetime` + `par`**, **`unsafe` + `mir_*`**, **`timetravel_borrow`**. |
| `demos/spacetime_collider.r4d` | Large **`list[vec4]`**, multiple **`spacetime:`** frames, **`Particle`** witness, narrative **`print`**. |
| `demos/causal_oracle.r4d` | **`ollama_demo()`** integration. |
| `demos/quantum_server.r4d` | **`quantum_server_demo()`** + env vars. |
| `demos/cosmic_genesis.r4d` | **10M** `list[vec4]`, **`par for`** rotor sweep, epic static transcript. |
| `libgeo.r4d` | Importable **`bump`**, **`identity_v4`**. |

**When asked for “the most advanced single file”:** start from **`spacetime_collider.r4d`** and **remove** features you cannot justify—not the other way around.

---

## 21. File extensions

| Ext | Status |
|-----|--------|
| **`.r4d`** | **Official** |
| **`.r4s`** | Legacy |
| **`.roma4d`** | Legacy |

---

## 22. Python vs Roma4D — invalid pattern encyclopedia

Each row is a **hard “do not generate”** unless the Roma4D compiler explicitly gains support and this guide is updated.

| Wrong pattern | Why | Use instead |
|---------------|-----|-------------|
| **`f"…{x}…"`** | No f-strings | **`print("literal")`**, separate prints |
| **`print(n)`** for **`int`/`float`** | Weak / non-portable runtime formatting | Fixed strings only |
| **`time()` / `sleep()`** | No such builtins | **`t`** token; external benchmarking via **`-bench`** |
| **`[T; N]`**, **`Vec4`**, **`float4`** | Wrong language | **`list[vec4]`**, **`vec4(...)`** |
| **`cosmos: [Particle; N] = …`** | Invalid | **`list[vec4]`** + **`Particle()`** witness if needed |
| **`import *`** | Hard error | **`from m import a, b`** |
| **`import os`, `import sys`, PyPI** | No linkage model | **`from libgeo import …`** or new **`.r4d`** module |
| **Nested `spacetime:` inside `for`** | Not in canonical demos | Sequential **`spacetime:`** blocks or outer structure |
| **`par for i in range(N): a[i] = …`** | Not the proven SIMD path | **`par for p in a: p = p * rot`** |
| **`async` / `await` / `lambda`** | Unsupported lowering story | **`def`** + straight-line code |
| **`try`/`except` as control flow** | Unstable for generators | Write type-safe code |
| **`open("f.txt")`**, **`requests`** | No builtins | C / future FFI (not here) |
| **`numpy` / `torch`** | Wrong runtime | **`list[vec4]`** + **`par`** |
| **`if __name__ == "__main__"`** | Meaningless | **`def main()`** only |
| **`# type: ignore`** | Meaningless | Fix the type |
| **Dynamic SQL / HTTP in `.r4d`** | No string runtime | **`ollama_demo`** / C |

---

## 23. Compiler and linker error catalog

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `source file not found` | Bad relative path / cwd | **Absolute path** to `.r4d` |
| `not a Roma4D source file` | Wrong extension | **`.r4d`** |
| `could not find Roma4D installation` / `roma4d.toml` | Outside tree + no env/embed | Run **`.\scripts\Install-R4dUserEnvironment.ps1`** once, new terminal, then **`r4d yourfile.r4d`** |
| `ImportError` | Missing module file | Add **`libfoo.r4d`** at correct path |
| `import * is not supported` | Python habit | Named imports |
| `UseAfterMoveError` | SoA linearity | Write-back **`cell.pos`** |
| `BorrowError` | Overlapping borrows | Narrow scope |
| `TaintError` | `print` → `soa` | Break the dataflow |
| `mm_malloc.h` / MinGW | Clang without Zig | **Install Zig** or **`R4D_GNU_ROOT`** |
| `zig cc` failed | Zig missing | **`PATH`** / **`R4D_ZIG`** |
| `synthesized return` | `-> int` without `return` | Add **`return 0`** |

---

## 24. Ergonomics: `r4`, `GOBIN`, `R4D_PKG_ROOT`, embedded root

- **`go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d`** puts binaries in **`go env GOBIN`** if set (e.g. conda **`…\Scripts`**), else **`GOPATH\bin`**.
- **`scripts/Install-R4dUserEnvironment.ps1`** (recommended on Windows): appends that **install directory** to **user** **`PATH`**, sets **`R4D_PKG_ROOT`**, optionally **`R4D_GNU_ROOT`** for Clang+MinGW, and runs **`go install`** with **`-ldflags -X …cli.EmbeddedPkgRoot=…`** so **`r4d C:\anywhere\sketch.r4d`** works after a new terminal.
- **`install.ps1`** / **`install.sh`**: same **embedded root** pattern for developers who **`go install`** from the repo.
- **`r4d.ps1`** / **`r4d.cmd`**: developer launcher at repo root—**`Set-Location $PSScriptRoot`**, **`go build`** into Go bin, prepend **`PATH`**, then **`& r4d.exe @args`**. No args → prints a short usage box and **`r4d help`**.
- Plain **`go install`** without **`-ldflags`**: **no embedded root**—either keep sources under a tree that contains **`roma4d.toml`** or set **`R4D_PKG_ROOT`** / **`ROMA4D_HOME`**.
- **CLI errors:** “file not found” suggests a **full path** example; “could not find Roma4D installation” points at **Install-R4dUserEnvironment.ps1** and **`r4d yourfile.r4d`**.

---

## 25. LLM code-generation protocol (step-by-step)

**Phase A — Requirements extraction**

1. Does the user need **exit code**? → **`main() -> int`** + **`return`**.
2. Otherwise → **`main() -> None`**.
3. Do they need **imported helpers**? → only **`from <existing_module> import …`** (`libgeo`).

**Phase B — Data model**

1. Bulk field in 4D? → **`list[vec4]`**.
2. Need SoA story? → **`class`** with **`soa pos: vec4`** + **`Particle()`** witness (see collider demo).
3. Pick **`N`**: **`1024`**, **`1_000_000`**, or (heavy) **`10_000_000`**.

**Phase C — Math**

1. Rotations → **`rotor(angle=…, plane="xy"|"yz"|"xw")`**.
2. Apply with **`*`** on **`vec4`**.

**Phase D — Parallelism**

1. Wrap **`par for`** in **`spacetime:`** like **`hello_4d.r4d`**.

**Phase E — Narrative I/O**

1. **`print("fixed line")`** only.

**Phase F — Self-verify**

1. Run **[§26 checklist](#26-pre-submit-checklist-copy-for-every-generation)**.

---

## 26. Pre-submit checklist (copy for every generation)

- [ ] Filename ends with **`.r4d`** (or legacy only if requested).
- [ ] **`def main()`** exists; no `__main__` guard.
- [ ] **Return discipline:** `-> int` always returns **`int`**; `-> None` uses **`return None`** or omits.
- [ ] **No f-strings**; **no `time()`**; **no `import *`**; **no PyPI**.
- [ ] **4D math** uses **`vec4` / `rotor` / `multivector`**, not Rust/C++ types.
- [ ] **SoA:** every **`cell.pos` read** paired with **write-back** before second read.
- [ ] **`par for`** over **`list`**, not index loop, unless verified.
- [ ] **`spacetime:`** structure matches **`hello_4d`** / **`spacetime_collider`** (no nested-in-`for` unless verified).
- [ ] **`print`** uses **string literals** only.
- [ ] **Imports** name real symbols from **`libgeo.r4d`** (or other existing files).
- [ ] Tell human to run **`r4d file.r4d`** and read **`debug/last_build_failure.log`** on failure.

---

## 27. LLM hard rules (non-negotiable)

1. Output **only** **`.r4d`** (unless user explicitly requests legacy extensions).
2. **Never** claim CPython semantics for **`print`**, **`str.format`**, **`time()`**, or **f-strings**.
3. **Never** invent **stdlib** modules beyond local **`.r4d`** files in the package root.
4. **Always** use **`vec4` / `rotor` / `multivector`** for geometric code—never **`Vec4` / `float4`**.
5. **`main() -> int`** must end with an **integer `return`** on all paths.
6. If unsure about a feature, **omit** it—**`print("ok")`** beats a hallucination.
7. After code, instruct: **`r4d <file>.r4d`**, **`--strict`** for CI, **`R4D_DEBUG=1`**, and **`last_build_failure.log`**.

---

## Appendix A — Drop-in safe template (verified shape)

Copy this when the user gives no extra constraints. It matches **`examples/hello_4d.r4d`** minus optional extras:

```roma4d
# sketch.r4d — minimal safe Roma4D scaffold (adjust N for memory)
from libgeo import bump, identity_v4

class Particle:
    soa pos: vec4
    soa vel: vec4

def main() -> None:
    _seed: int = bump(0)
    rot: rotor = rotor(angle=1.5707963267948966, plane="xy")
    worldtube: list[vec4] = [vec4(x=0, y=0, z=0, w=1) for _ in range(1024)]

    _tau: time = t
    w: vec4 = worldtube[0] @ t
    _ = identity_v4(w)

    cell: Particle = Particle()
    col: vec4 = cell.pos
    col = col * rot
    cell.pos = vec4(x=0, y=0, z=0, w=1)

    spacetime:
        par for p in worldtube:
            p = p * rot

    print("ok")
    return None
```

---

## 28. Document maintenance

When the compiler changes, update:

| Area | File(s) |
|------|---------|
| Builtins | `src/compiler/typechecker.go` (`seedBuiltins`) |
| Extensions / resolution | `src/compiler/source_ext.go` |
| Native AI Expert | `src/ai/expert.go` (terminal debug, interactive, LLM briefing) |
| Keywords | `src/parser/token.go` |
| CLI / pkg root | `internal/cli/cli.go` |
| Windows install + PATH | `scripts/Install-R4dUserEnvironment.ps1`, `r4d.ps1`, `r4d.cmd` |
| Linker | `src/compiler/llvm_link.go` |
| Void main ABI | `src/compiler/codegen_llvm.go` |
| Runtime | `rt/roma4d_rt.c` |
| **This guide** | `docs/Roma4D_Guide.md` |

---

## 29. Native AI Expert (terminal debug + interactive)

**What it is:** A **built-in**, **zero-network** assistant that runs when **`r4d` / `roma4d` / `r4`** fails to compile or link a **`.r4d`** file in **forgiving mode** (the default). It does **not** call external LLMs; it uses curated rules aligned with this guide.

### 29.1 Terminal debug block (always in forgiving mode)

On failure, stderr receives a bordered block you can **copy in full** for humans or for an external LLM. It includes:

| Section | Contents |
|--------|-----------|
| Header / footer | `ROMA4D EXPERT DEBUG` lines (easy grep / paste boundaries) |
| Metadata | UTC time, `GOOS`/`GOARCH`, Go version, tool name, action, source path, package root (if known), inferred line |
| Raw message | Verbatim compiler / linker / driver text |
| Source context | A few lines around the inferred error line (`bufio.Scanner` on the source file) |
| Symptom hints | Short bullets mapping this failure to guide sections (e.g. SoA §11, Zig §19) |
| Copy/paste commands | Examples: `r4d C:\full\path\file.r4d`, `R4D_DEBUG=1`, `winget install Zig.Zig`, `pip install PyQt5` when relevant |
| Guide memory | Pointers to the normative sections of **this document** |
| **LLM_INSTRUCTIONS** | A ready-to-paste briefing telling an assistant to output only valid **`.r4d`** per §27 / §22 |

**Strict mode:** `r4d --strict ...` skips the entire Expert path and prints a minimal one-line error.

### 29.2 Log file

The same session is **appended** to **`debug/last_build_failure.log`** under the **package root** (via `compiler.WriteBuildFailureLog`, stage `ai_expert`). Structured linker failures from the driver may also append earlier rows for the same run.

### 29.3 Interactive Expert (TTY only)

If **stdin is a terminal** (detected with `golang.org/x/term`), after the debug block the Expert may:

1. Show a **suggested edit** for high-confidence cases (e.g. `import *` → explicit imports) and prompt **Done? [y/N]** (acknowledgment — you apply the edit in your editor).
2. Open a small **REPL**: type **`why`**, **`zig`**, **`pyqt`**, **`guide`**, **`llm`**, **`strict`**, **`help`**, or **`quit`**.

**Disable interactive mode** (keep the printed debug block): set **`R4D_EXPERT_INTERACTIVE=0`**.

### 29.4 Implementation rules (for contributors)

- **Regex:** All patterns in `expert.go` use **`regexp.MustCompile`** at package scope.
- **File reads:** Source context uses **`bufio.Scanner`** with an explicit buffer cap (no unbounded slurp of huge files).
- **Panic safety:** The Expert **`defer recover()`** fail-opens to a short raw error line if something unexpected panics.
- **Python vs Roma4D:** The Expert reminds users that **`.py`** is delegated to CPython via **`r4d script.py`**; **`.r4d`** stays on the native pipeline (see CLI help).

### 29.5 Quick reference for users

| Goal | Command / action |
|------|-------------------|
| Rich failure help | Run without `--strict` (default). |
| CI / scripts only | `r4d --strict file.r4d` |
| Skip post-mortem prompts | `R4D_EXPERT_INTERACTIVE=0` |
| Mirror linker details | `R4D_DEBUG=1` (see §19) |
| Feed an external LLM | Copy from **`LLM_INSTRUCTIONS`** through **`END_LLM_INSTRUCTIONS`** in the debug block |

---

## 30. Spacetime engine foundations (theory manuscript)

This section **deciphers and restates** companion material (PowerShell deployment snippets, Python demos, and the “Roma4D manuscript”) in **professional technical language**. It is **descriptive**: it explains **intent and mathematics**. It does **not** override [§9 Geometric algebra](#9-geometric-algebra-cl40-operators) or the **normative rule** that **`.r4d`** is the official Roma4D program form—not arbitrary Python.

### 30.1 What the deployment snippets actually do

The quoted **`Deploy_Roma4D.ps1` / `Deploy_Roma4D_Quantum.ps1`** pattern:

1. Creates a folder such as `...\RomanAILabs\4d_Spacetime_engine`.
2. Writes a **standalone Python** file (`Roma4D_Tesseract.py`, `Roma4D_Hyper_Quantum_Core.py`).
3. Runs it with **`python`**.

**Interpretation:** these are **pedagogical / lab demos** (ASCII projection of a tesseract; NumPy toy for “isoclinic” coordinate rotations and a mock attention-to-lattice bump). They are **not** the Roma4D **AOT compiler**, **not** `r4d` linking LLVM output, and **not** a substitute for **`.r4d`**.

| Artifact | Technique in code | Relation to Cl(4,0) in this guide |
|----------|-------------------|-------------------------------------|
| **Tesseract script** | Stereographic map \(\mathbb{R}^4 \to \mathbb{R}^3\), then pinhole-style \(\mathbb{R}^3 \to\) 2D grid; **XW** rotation as a **linear 2×2** mix of \(x,w\) | **Motivation** for 4D geometry; **does not** implement the full **16-component** multivector API |
| **Hyper Quantum Core** | Independent **XY** and **ZW** plane rotations on \((x,y,z,w)\); random **NumPy** “\(Q,K\)” mats; Frobenius norm scales a lattice cell | **Illustrative** coupling story; **not** live Ollama tensor hooks unless you build that integration |

### 30.2 Mathematical substrate (RomanAILabs narrative, formalized)

**Clifford algebra \(Cl(4,0)\).**  
Euclidean signature: \(e_i^2 = 1\), \(e_i e_j = -e_j e_i\) for \(i \neq j\). A **point / direction** is modeled as a **grade-1 vector** \(P = x e_1 + y e_2 + z e_3 + w e_4\).

**Rotors.**  
A rotation in the plane generated by bivector \(B\) (e.g. \(e_1 e_4\) for an **XW** plane) uses the even element  
\(R = \exp(-\tfrac{\theta}{2} B)\), and for simple unit bivectors in this signature the sandwich \(P' = R P \tilde{R}\) matches the usual **SO(4)** action. This avoids Euler-angle gimbal issues in **4D** and unifies many transforms as **one** algebraic product.

**Fourth dimension \(w\) vs engine time.**  
In the manuscript, **\(w\)** is treated as a **spatial** fourth axis; **simulation time** or **engine ticks** are **external** to \((x,y,z,w)\). That yields a useful mental model: **4 spatial dimensions + 1 execution clock** (a **4+1** *modeling* choice, not a claim about physical relativity unless you explicitly wire Lorentzian metric elsewhere).

**4D lattice waves.**  
A **scalar field** \(\phi\) on a 4D grid with discrete Laplacian  
\(\nabla^2 \phi = \delta_{xx}\phi + \delta_{yy}\phi + \delta_{zz}\phi + \delta_{ww}\phi\)  
and a **leapfrog / second-order** time discretization is a standard explicit PDE pattern for **hypercube** neighborhoods. The manuscript uses it as the **propagation** metaphor for disturbances and interference.

### 30.3 “Q-tensor” and bidirectional coupling (vision layer)

The manuscript proposes **coupling** between:

- **Lattice / field** quantities (e.g. gradients or \(\phi\) on the **\(w\)** axis), and  
- **Neural inference** (e.g. local LLM attention statistics),

including a **reverse** path: norms derived from attention-like matrices feed a **source term** in a wave or diffusion picture, and **Hodge duality** in \(Cl(4,0)\) maps bivector planes to orthogonal planes (e.g. **\(xy\)** vs **\(zw\)** in the split you choose).

**Professional reading:** these are **research directions** and **architecture metaphors**. Shipping them requires: a defined **API** (e.g. HTTP to Ollama), **numerical stability** analysis for nonlinear sources, and **traceable telemetry**—not just narrative.

### 30.4 Theoretical horizons (explicitly speculative)

Items such as “hyper-dimensional memory crystals,” “4D cryptographic hiding in \(w\),” “temporal dilation of inference,” and “multi-model gravity wells” are **speculative product theses**. Treat them as **design prompts** for future operators and demos, **not** as guarantees of the current compiler or runtime.

### 30.5 How LLMs should use this section

- **Normative codegen** for Roma4D: still **§9**, **§11**, **§27** (`.r4d` only).  
- **This §30**: use when the user asks for **4D GA meaning**, **manuscript alignment**, or **Python demo provenance**—and always **separate** “vision math” from “what `r4d` accepts today.”

---

*End of Roma4D Elite Reference Guide.*
