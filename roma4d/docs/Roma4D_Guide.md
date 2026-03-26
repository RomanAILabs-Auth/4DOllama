# Roma4D Programming Guide

**Purpose:** This document is the **authoritative reference** for **Roma4D** sources. Official extension is **`.r4d`**. Legacy **`.r4s`** and **`.roma4d`** remain accepted. Use it for **humans** and **LLMs** — follow it literally to avoid invalid “Python disguised as Roma4D.”

**Toolchain version:** `0.1.0` (see `roma4d.toml` and `internal/cli/cli.go`).  
**Language edition:** `2025` (manifest field; describes the staged feature set, not calendar year alone).

**Not Python:** Roma4D uses **Python 3.12–shaped** syntax (indentation, `def`, `class`, many keywords) but is a **separate, statically compiled** language. Do not assume arbitrary Python stdlib, dynamic `eval`, or full PEP compatibility.

---

## Table of contents

- [Mental Model (Read This First)](#mental-model-read-this-first)

1. [How the compiler runs](#1-how-the-compiler-runs)
2. [Project layout and `roma4d.toml`](#2-project-layout-and-roma4dtoml)
3. [CLI commands](#3-cli-commands)
4. [Modules and imports](#4-modules-and-imports)
5. [Entry point and functions](#5-entry-point-and-functions)
6. [Types](#6-types)
7. [Builtins and constructors](#7-builtins-and-constructors)
8. [Geometric algebra (Cl(4,0)) operators](#8-geometric-algebra-cl40-operators)
9. [Classes, `soa`, and `aos`](#9-classes-soa-and-aos)
10. [Ownership 2.0 (linear SoA, borrows)](#10-ownership-20-linear-soa-borrows)
11. [Spacetime: `t`, `@ t`, `spacetime:`](#11-spacetime-t--t-spacetime)
12. [Parallelism: `par for`](#12-parallelism-par-for)
13. [Systems: `unsafe:` and MIR hooks](#13-systems-unsafe-and-mir-hooks)
14. [Native runtime (`rt/roma4d_rt.c`)](#14-native-runtime-rtroma4d_rtc)
15. [Ollama / HTTP demos (builtins)](#15-ollama--http-demos-builtins)
16. [Compilation pipeline (mental model)](#16-compilation-pipeline-mental-model)
17. [Help, debugging, and common failures](#17-help-debugging-and-common-failures)
18. [LLM checklist (generate valid Roma4D)](#18-llm-checklist-generate-valid-roma4d)
19. [Example programs in this repo](#19-example-programs-in-this-repo)
20. [File extensions: `.r4d` and legacy `.r4s` / `.roma4d`](#20-file-extensions-r4d-and-legacy-r4s--roma4d)
21. [Python vs Roma4D — invalid patterns (do not generate)](#21-python-vs-roma4d--invalid-patterns-do-not-generate)
22. [Compiler and linker error catalog](#22-compiler-and-linker-error-catalog)
23. [Ergonomics: `r4`, PATH, `R4D_PKG_ROOT`](#23-ergonomics-r4-path-r4d_pkg_root)
24. [LLM hard rules (non-negotiable)](#24-llm-hard-rules-non-negotiable)

## Mental Model (Read This First)

Roma4D is built around three core ideas:

1. **Columns, not objects**  
   Data lives in Structure-of-Arrays (SoA) by default — not pointer-heavy objects.

2. **Time is explicit**  
   `t`, `@ t`, and `spacetime:` blocks make *when* something happens a first-class concept.

3. **Execution is parallel by design**  
   `par for` is the natural way to express large-scale work.

If you think in **columns of data**, **time-indexed values**, and **parallel transformations**, you will write fast, natural, and idiomatic Roma4D code.

---

## 1. How the compiler runs

- **Inputs:** One **`.r4d`** (or legacy **`.r4s`** / **`.roma4d`**) source file per build/run, plus any imported modules under the **package root** (directory containing `roma4d.toml`). Imports resolve to **`name.r4d` first**, then **`.r4s`**, then **`.roma4d`**.
- **Pipeline:** `lexer → parser → typecheck → Ownership 2.0 pass → MIR → LLVM IR` → **Windows:** `zig cc` (default) or `clang` + MinGW; **Unix:** `clang` (compile + link).
- **Output:** Native executable (Windows: `.exe`). Link step may compile and link `rt/roma4d_rt.c` for runtime symbols (`print`, `vec4`, `Particle`, etc.).
- **Backend:** `[build] backend = "llvm"` in `roma4d.toml`. CUDA / GPU paths are roadmap; some MIR metadata exists for GPU-oriented `spacetime` + `par` regions.

**Package root:** The driver walks **upward** from the source file’s directory until it finds `roma4d.toml`. If none is found, it uses **`R4D_PKG_ROOT`** or **`ROMA4D_HOME`** when that directory contains `roma4d.toml` (see §3 — run sketches from anywhere). That resolved directory is **`pkgRoot`**.

---

## 2. Project layout and `roma4d.toml`

Typical layout (this repository):

```text
roma4d/
  roma4d.toml          # required manifest
  cmd/r4d/             # CLI
  src/compiler/        # typecheck, MIR, LLVM, link driver
  src/parser/          # lexer + parser
  rt/roma4d_rt.c       # C runtime linked into user programs
  libgeo.r4d           # example library module (official .r4d)
  examples/*.r4d
  demos/*.r4d
  debug/               # build failure logs (generated)
```

**`roma4d.toml` (authoritative fields used today):**

- `name` — logical package name; contributes to **qualified module names** for symbols.
- `version`, `edition` — metadata.
- `[package]` — description, authors.
- `[build]` — `default_backend`, `backend`, `incremental`.
- `[systems]` — `gc = false`, `unsafe = true` (systems features allowed at language level).

---

## 3. CLI commands

**Binaries:** **`r4`** (short), **`r4d`**, and **`roma4d`** share one implementation (`internal/cli`). Prefer **`r4`** for day-to-day use — same flags as `r4d`.

**Forgiving Expert mode (default):** On compile failure, the CLI may append **native** Roma4D hints (rules in `src/ai/expert.go`) — no external LLM, no network. Use **`--strict`** anywhere in the argv to print **raw** errors only (CI, scripting).

| Command | Meaning |
|--------|---------|
| `r4 help` / `r4d help` | Usage text. |
| `r4 version` | Prints `roma4d (r4d) <ver> <os>/<arch>` (banner text unchanged). |
| `r4d <file.r4d> [args...]` | Shorthand for **`r4d run`** (same Expert / `--strict` behavior). |
| `r4 build [--strict] <file.r4d> [-o path] [-bench]` | Compile to a native executable. Basename strips **`.r4d`** / **`.r4s`** / **`.roma4d`**. |
| `r4 run [--strict] <file.r4d> [-bench] [args...]` | Temp build + run. |

**`-bench`:** Phase timings (`load_manifest`, `parse`, `typecheck`, `lower_*`, `zig_*` or `clang_*`, and `native_run` for `run`).

**PowerShell note (Windows):** When pasting commands, **do not** paste the `PS C:\...>` prompt. On some systems `PS` is an alias and breaks the line.

**Recommended Windows workflow:** From `roma4d/`, use **`.\r4d.ps1`** — it rebuilds **`r4.exe`**, **`r4d.exe`**, **`roma4d.exe`** into `GOPATH\bin` and runs **`r4`** with your args.

### Run from any directory (Windows)

1. **One-time:** From the `roma4d/` repo root:

   ```powershell
   .\scripts\Install-R4dUserEnvironment.ps1
   ```

   **`go install`s** `r4`, `r4d`, `roma4d`; appends **`%GOPATH%\bin`** to **user** `PATH`; sets **`R4D_PKG_ROOT`**.

2. **Open a new PowerShell window.**

3. **Examples:**

   ```powershell
   r4 run C:\Users\You\Desktop\4DEngine\roma4d\demos\cosmic_genesis.r4d
   r4 run C:\Users\You\Desktop\my_sketch.r4d
   ```

   Imports (`from libgeo import ...`) resolve under **`R4D_PKG_ROOT`**, not next to the sketch.

**Environment variables (any OS):**

| Variable | Purpose |
|----------|---------|
| `R4D_PKG_ROOT` | Fallback package root if `roma4d.toml` is not found above the source file. |
| `ROMA4D_HOME` | Same as `R4D_PKG_ROOT`. |

**Relative paths:** `r4 run demos\foo.r4d` is resolved from **cwd**. Wrong directory → “file not found”; use a **full path** or `cd` to the repo first.

---

## 4. Modules and imports

### 4.1 Module file resolution

Given `from foo.bar import x` or `import foo.bar`, the checker resolves **`foo.bar`** to a file under **package root**. For each candidate path it tries **`.r4d` first**, then **`.r4s`**, then **`.roma4d`**:

1. `pkgRoot/foo/bar.r4d`, `pkgRoot/foo/bar.r4s`, or `pkgRoot/foo/bar.roma4d`
2. `pkgRoot/foo.bar.r4d`, `pkgRoot/foo.bar.r4s`, or `pkgRoot/foo.bar.roma4d`

If none exist → **ImportError**.

### 4.2 `import` forms

- **`import mod`** — binds module name `mod` (or alias with `as`).
- **`from mod import a, b`** — imports exported names from module `mod`.
- **`import *`** — **not supported** (hard error in the typechecker).

### 4.3 What a module exports (current compiler)

When loading a submodule, the compiler collects **top-level** `def` and `class` names into `Exports`. It does **not** fully type-check the submodule body in the same pass as a full second compilation unit (v0 behavior); still, **only names present as top-level defs/classes** are importable.

**Practical rule:** Put shared functions and classes at **module top level** in `libfoo.r4d` (or legacy `libfoo.r4s` / `libfoo.roma4d`).

### 4.4 Example

`libgeo.r4d` at package root:

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

## 5. Entry point and functions

- **`def main()`** is the program entry point.
- **Return type:** Annotate as `-> None` or `-> int` (etc.). If you omit return on non-`None`, the compiler may **synthesize a return** (warning). Prefer explicit `return` for clarity.
- **Type annotations:** Use on parameters and return types where possible; the typechecker relies on them for defs and class fields.
- **Statements:** Python-like `if` / `else`, `for`, `while`, `return`, `pass`, assignment, augmented assignment, expression statements.

---

## 6. Types

### 6.1 Core primitive / host types

| Type | Notes |
|------|--------|
| `int` | Signed integer; used by `mir_ptr_load` / `mir_ptr_store` bridge. |
| `float` | Floating point. |
| `str` | String type; **`print` accepts `str`**. |
| `bool` | `True` / `False`. |
| `none` | `None`; also used as “no result” for some builtins. |

### 6.2 Native 4D / systems types

| Type | Role |
|------|------|
| `vec4` | 4D vector (Cl(4,0) reference numerics in codegen). |
| `rotor` | Rotor (plane + angle constructor in surface language). |
| `multivector` | General multivector. |
| `rawptr` | Raw pointer for `unsafe:` / MIR heap ops. |
| `time` | Type of **`t`** (compile-time temporal coordinate token). |

**Qualified module names:** Derived from `roma4d.toml` `name` + path relative to package root (see `qualifyModule` in `typechecker.go`).

---

## 7. Builtins and constructors

These are **predefined** in the typechecker (`seedBuiltins`). Names are case-sensitive.

| Name | Arity / form | Result (approx.) |
|------|----------------|------------------|
| `print` | Variadic | `none` — prints strings (runtime: `puts`). |
| `range` | Variadic | `list[int]`-like iterable for `for` / comprehensions. |
| `len` | `(any)` | `int` |
| `int`, `float`, `str`, `bool` | Constructors / casts (variadic surface) | respective type |
| `abs` | `(any)` | `any` (typed loosely in v0) |
| `vec4` | Keyword args, e.g. `vec4(x=0, y=0, z=0, w=1)` | `vec4` |
| `rotor` | e.g. `rotor(angle=..., plane="xy")` | `rotor` |
| `multivector` | `multivector()` | `multivector` |
| `borrow` | `(x)` | borrow marker (ownership); argument **must** be a **simple name** for the borrow bookkeeping. |
| `mutborrow` | `(x)` | mutable borrow; same name restriction pattern. |
| `timetravel_borrow` | `(x)` | chronology borrow; ownership same as `borrow` at pass level. |
| `mir_alloc` | `(size: int)` | `rawptr` |
| `mir_ptr_store` | `(ptr: rawptr, value: int)` | `none` |
| `mir_ptr_load` | `(ptr: rawptr)` | `int` |
| `ollama_demo` | `()` | `int` — runs **curl** to local Ollama (see §15). |
| `quantum_server_demo` | `()` | `int` — quantum snapshot + Ollama (see §15). |
| `True`, `False`, `None` | literals | `bool` / `none` |

**Important:** There is **no** general dynamic string runtime for building HTTP bodies inside **`.r4d`** alone; demos that need JSON use **C + curl** in `roma4d_rt.c`.

---

## 8. Geometric algebra (Cl(4,0)) operators

On **`vec4`**, **`rotor`**, **`multivector`** (where implemented), these operators mean **geometric** product / outer product / contraction:

| Operator | Meaning (4D types) | Contrast (plain ints) |
|----------|-------------------|------------------------|
| `*` | Geometric product (e.g. `vec4 * rotor`) | Integer multiply |
| `^` | Outer product | Integer **XOR** |
| `|` | Contraction / inner-style product | Integer **OR** |

**Disambiguation:** The parser/typechecker use **types** to choose algebra vs bitwise. Mixing incorrectly produces type errors or wrong lowering.

**Example pattern:**

```roma4d
a: vec4 = pos[0] * rot
b: multivector = pos[0] ^ mv
c: float = pos[0] | mv
i: int = 3 ^ 5
j: int = 1 | 2
```

---

## 9. Classes, `soa`, and `aos`

```roma4d
class Particle:
    soa pos: vec4
    soa vel: vec4
```

- **`soa`** marks **structure-of-arrays** / column-style fields with **linear** access semantics (see §10).
- **`aos`** exists as a keyword in the lexer for row-style layout hints; check current passes for full enforcement — **`soa` is the well-trodden path** in examples.

**Constructing:** `Particle()` lowers to a runtime constructor (`Particle` symbol in `roma4d_rt.c`).

---

## 10. Ownership 2.0 (linear SoA, borrows)

**Goal:** Make aliasing and **parallel** (`par`) safety explicit. The **native binary does not run a borrow checker** — all checks are **compile-time**.

### 10.1 Linear `soa` field access

Pattern (from `examples/hello_4d.r4d`):

```roma4d
cell: Particle = Particle()
col: vec4 = cell.pos      # move out of slot
col = col * rot
cell.pos = vec4(x=0, y=0, z=0, w=1)   # write back before next read
again: vec4 = cell.pos
```

**Errors you may see:**

- **`UseAfterMoveError`** — read `soa` field again without re-assigning.
- **`BorrowError` / `UseError`** — moved or used while `borrow` / `mutborrow` active.
- **`TaintError`** — assigning a **Python-tainted** value (e.g. flow after `print`) into a **linear / `soa` slot**.

**LLM rule:** After **every** `cell.pos` read into a local, **reassign** `cell.pos` before another read, unless your dataflow proves a single consume.

### 10.2 `borrow`, `mutborrow`, `timetravel_borrow`

- **`borrow(x)`** — immutable borrow of name `x` (must be a **simple identifier** in the borrow capture logic).
- **`mutborrow(x)`** — exclusive mutable borrow; cannot coexist with immutable borrows.
- **`timetravel_borrow(x)`** — same **ownership** discipline as `borrow` at this pass; MIR distinguishes it for **chrono / spacetime** story.

Borrows are **scoped to lexical blocks** and to **distinct `spacetime:` regions** (compile-time epoch labels in diagnostics).

---

## 11. Spacetime: `t`, `@ t`, `spacetime:`

**Compile-time staging:** `t`, `expr @ t`, and `spacetime:` blocks participate in **static** reasoning and MIR metadata. They **do not** introduce a temporal interpreter loop in the emitted binary today.

```roma4d
_tau: time = t
sample: vec4 = worldtube[0] @ t
```

```roma4d
spacetime:
    par for p in worldtube:
        p = p * rot
    _hold = timetravel_borrow(rot)
```

**LLM guidance:** Use **`spacetime:`** to group **physics narrative** and **parallel** regions; keep heavy work in **`par for`** over **`list[vec4]`** or SoA columns.

---

## 12. Parallelism: `par for`

```roma4d
spacetime:
    par for p in positions:
        p = p * rot
```

**Semantics (today):**

- Marks a **structured parallel** loop for **sendability** / capture checking and **backend hints**.
- **SIMD-friendly** lowering exists for some geometric ops (e.g. `vec4` with `rotor`).
- **GPU / CUDA** full path is not the default end-to-end experience; linking may involve stubs when GPU metadata is present — consult MIR/LLVM for flags.

**Capture rule:** The ownership pass collects names used inside `par` bodies; **borrows** must not accidentally capture forbidden state. Treat `par` bodies as **side-effect disciplined** — deterministic updates to loop variables.

---

## 13. Systems: `unsafe:` and MIR hooks

```roma4d
unsafe:
    rawp: rawptr = mir_alloc(128)
    mir_ptr_store(rawp, 42)
    _peek: int = mir_ptr_load(rawp)
```

- **`unsafe:`** opens a **systems** region (manifest `unsafe = true`).
- **`mir_alloc` / `mir_ptr_store` / `mir_ptr_load`** lower to calls backed by **malloc** + load/store in LLVM codegen.

**No GC:** `[systems] gc = false` — memory management is manual / arena-style at user discretion.

---

## 14. Native runtime (`rt/roma4d_rt.c`)

Linked when present at `pkgRoot/rt/roma4d_rt.c`. Provides C symbols expected by LLVM codegen, including:

- **`print`** → `puts`
- **`vec4`**, **`rotor`**, **`multivector`**, **`Particle`**
- **`bump`**, **`identity_v4`** — if you add matching defs, prefer **Roma4D `lib*.r4d`** for source-level libraries; C duplicates are for ABI demos only.
- **`identity_v4`**, geometric stubs as needed for lowering
- **`ollama_demo`**, **`quantum_server_demo`** — shell out to **`curl`** (see §15)

**Float formatting:** Native `print` is **string-oriented**; numeric `printf` from Roma4d is not the general story yet — demos use **fixed strings** or C-side output for detailed numbers.

---

## 15. Ollama / HTTP demos (builtins)

### 15.1 `ollama_demo()`

- **Requires:** `curl` on `PATH`, local **Ollama** (`ollama serve`, default `http://127.0.0.1:11434`), model pulled (e.g. `qwen2.5`).
- **Behavior:** Fixed JSON body embedded in C (because dynamic host strings are not generally available in **`.r4d`** yet).
- **See:** `demos/causal_oracle.r4d`.

### 15.2 `quantum_server_demo()`

- **Requires:** Same as above.
- **Environment variables:**
  - **`QUANTUM_QUERY`** — optional user question (sanitized for JSON).
  - **`QUANTUM_CONTINUE=1`** — load/save **4-qubit statevector** snapshot file under `TEMP` / `TMPDIR` for **cross-run** continuity.
- **See:** `demos/quantum_server.r4d`.

**Security note:** These call **`system("curl ...")`** from C — only as trusted as your local shell and paths.

---

## 16. Compilation pipeline (mental model)

1. **Parse** `.r4d` / `.r4s` / `.roma4d` → AST (`src/parser`).
2. **Typecheck** → types on nodes; imports resolved; builtins known (`src/compiler/typechecker.go`).
3. **Ownership 2.0** → linear SoA + borrow conflicts (`src/compiler/ownership.go`).
4. **Lower to MIR** (`src/compiler/ast_to_mir.go`).
5. **LLVM IR** (`src/compiler/codegen_llvm.go` + helpers).
6. **Clang:**
   - Compile `.ll` → `.o`
   - Link `.o` + `roma4d_rt.c` (+ optional CUDA stub) → executable  
7. **Windows:** **`zig cc -target *-windows-gnu`** (default) or **clang** with the same triple → **MinGW** ABI, not MSVC by default.
8. **Unix:** linker adds **`-lm`** for math in `roma4d_rt.c`.

---

## 17. Help, debugging, and common failures

### 17.1 Build failure log (always check this first)

On failure, the driver appends a structured record to:

**`pkgRoot/debug/last_build_failure.log`**

It includes **stage** (`zig_compile`, `zig_link`, `clang_compile`, `clang_link`, etc.), **full tool argv**, **stderr**, and a **head slice of LLVM IR** for context.

### 17.2 `R4D_DEBUG`

```bash
export R4D_DEBUG=1    # Unix
set R4D_DEBUG=1       # Windows cmd
$env:R4D_DEBUG="1"    # PowerShell
```

Mirrors the same failure block to **stderr** immediately.

### 17.3 “`roma4d.toml` not found”

- Pass a **full path** to a **`.r4d`** file that lives **under** a directory tree containing `roma4d.toml`, **or**
- Set **`R4D_PKG_ROOT`** (or **`ROMA4D_HOME`**) to the folder that **contains** `roma4d.toml`, then run **`r4 run C:\anywhere\sketch.r4d`**.
- **Windows:** Run **`.\scripts\Install-R4dUserEnvironment.ps1`** once (see §3) to set user **`R4D_PKG_ROOT`** and put **`r4d`** on PATH.

### 17.4 Zig (Windows default) and Clang

- **Windows (recommended):** Install [**Zig**](https://ziglang.org/download/) and ensure **`zig` / `zig.exe`** is on `PATH`. Roma4D runs **`zig cc`** to compile `.ll` and link **`roma4d_rt.c`** (no separate MinGW install required).
- **Override:** set **`R4D_ZIG`** to the full path of the `zig` executable if it is not on `PATH`.
- **Windows fallback:** if Zig is missing, install **LLVM** and put **`clang`** on `PATH`, plus **MinGW-w64** (see §17.5).
- **Linux/macOS:** **`clang`** on `PATH` (`clang`, `clang-18`, … `clang.exe` on Windows fallback).

### 17.5 Windows: MinGW / `mm_malloc.h` / link errors (Clang fallback only)

**When:** You are on **Clang fallback** (Zig not installed). **Symptom:** Clang includes **`msys64/ucrt64/include/stdlib.h`** but fails on **`mm_malloc.h`**.

**Fix (automatic for clang):** The driver adds **`--gcc-toolchain=...`** and **`-isystem`** for GCC builtin and MinGW **`include`**. Easiest alternative: **install Zig** (§17.4) and avoid this path.

**Prefer Zig on Windows** so you do not need MSYS2 for normal builds.

**Override / non-default install:**

```powershell
$env:R4D_GNU_ROOT = "C:\msys64\ucrt64"
```

Ensure **MinGW `bin`** is on `PATH` so **ld** and CRT libs resolve.

**Hint text from tool:** Prefer **MinGW-w64** with **`clang -target *-windows-gnu`**, not MSVC, unless you change the driver.

### 17.6 Linux: undefined reference to `sqrt`

The driver passes **`-lm`** on non-Windows when linking `roma4d_rt.c`. If you customize link steps, preserve **`-lm`**.

### 17.7 Type / ownership errors

- Read the **exact** diagnostic (`UseAfterMoveError`, `BorrowError`, `ImportError`, etc.).
- Cross-check §4 (imports), §10 (SoA), §11–12 (`spacetime` / `par`).

### 17.8 Warnings: “synthesized return”

- **`def main() -> int:`** — use explicit **`return 42`** (etc.) on all paths; otherwise the compiler may synthesize a return and warn.
- **`def main() -> None:`** — you may omit `return`, use bare control flow, or **`return None`** (lowers to a void return; **do not** expect a Python `None` object at runtime).

### 17.9 Parser / indentation errors

Roma4D uses **indentation-based** blocks like Python. **Tabs vs spaces:** stay consistent; mixed indentation can confuse the lexer structure.

---

## 18. LLM checklist (generate valid Roma4D)

When asked to write Roma4D, follow this checklist **in order**:

1. **File extension:** **`.r4d`** (official). Legacy **`.r4s`** / **`.roma4d`** only if the user explicitly wants them.
2. **Package root:** `roma4d.toml` above the file **or** user has **`R4D_PKG_ROOT`** set. Imports resolve from that root (`libgeo.r4d`, etc.).
3. **Imports:** **`from mod import a, b`** only — **never** `import *` (hard error).
4. **`main`:** `def main() -> None:` or `-> int:`. For `int`, **always** `return` an integer. For `None`, `return None` is allowed and lowers to void.
5. **4D math:** **`vec4`**, **`rotor`**, **`multivector`**; `* ^ |` are **geometric** on those types — **not** Python `*` / `^` / `|` on ints by mistake inside GA expressions.
6. **SoA:** Read `soa` field → mutate → **assign back** before reading again.
7. **`par for`:** Prefer inside **`spacetime:`** (matches demos and MIR metadata).
8. **`unsafe:`** only for **`mir_alloc` / `mir_ptr_*`** and C-linked demos.
9. **`print`:** **string literals** (and what the runtime supports). **No** `print(f"...")`, **no** `print` of arbitrary formatted floats unless you add C support.
10. **Ollama demos:** **`ollama_demo()`**, **`quantum_server_demo()`** — document **curl**, **Ollama**, env vars (**`QUANTUM_QUERY`**, etc.).
11. **Run command:** **`r4d path/to/file.r4d`** or **`r4 run path/to/file.r4d`**; use **`--strict`** for raw errors only. Failures → **`debug/last_build_failure.log`**, **`R4D_DEBUG=1`**.

**Safe boilerplate (copy-paste safe):**

```text
# file: sketch.r4d
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

    spacetime:
        par for p in worldtube:
            p = p * rot

    print("ok")
    return None
```

---

## 19. Example programs in this repo

| Path | Role |
|------|------|
| `examples/min_main.r4d` | Smallest native `main` → `int` (**exit 42**). |
| `examples/hello_4d.r4d` | Full **4D + SoA + spacetime + par + unsafe** tour. |
| `demos/spacetime_collider.r4d` | Large **worldtube** + narrative **frames**. |
| `demos/causal_oracle.r4d` | **`ollama_demo()`** + spacetime story. |
| `demos/quantum_server.r4d` | **`quantum_server_demo()`** + **QUANTUM_*** env vars. |
| `demos/cosmic_genesis.r4d` | Shorter **worldtube + par** demo. |
| `libgeo.r4d` | Importable library (`bump`, `identity_v4`). |

---

## 20. File extensions: `.r4d` and legacy `.r4s` / `.roma4d`

| Extension | Status |
|-----------|--------|
| **`.r4d`** | **Official.** Use in all new files and docs. |
| **`.r4s`** | **Legacy, still accepted** (short suffix). |
| **`.roma4d`** | **Legacy, still accepted** (original long suffix). |

**Module resolution order:** For import `foo.bar`, the driver tries **`foo/bar.r4d`**, then **`foo/bar.r4s`**, then **`foo/bar.roma4d`**, then dotted-path variants — see `ResolveRoma4DModuleFile` in `src/compiler/source_ext.go`.

**Wrong extensions:** `.py`, `.rs`, `.c` are **not** Roma4D sources — the CLI rejects them with `not a Roma4D source file`.

---

## 21. Python vs Roma4D — invalid patterns (do not generate)

LLMs often emit **Python** by reflex. The following are **wrong in Roma4D** unless explicitly stated otherwise in *this* guide:

| Wrong (Python-ism) | Why it fails | Correct direction |
|--------------------|--------------|-------------------|
| `f"tick {x}"` / any **f-string** | Not the string model | Fixed **`print("literal")`** only |
| `print(x)` with **`float` / `int`** | Runtime is `puts`-style | Format in C or print fixed text |
| **`[Vec4; N]`** / **`Vec4(...)`** | Rust / C++ syntax | **`list[vec4]`**, **`vec4(x=..., y=..., z=..., w=...)`** |
| **`time()`** / **`elapsed = time() - start`** | No such runtime | **`time`** is **`t`** (compile-time token), not a callable |
| **Nested `spacetime:`** inside `for` | Not the demo / MIR shape | One outer **`spacetime:`**, **`par for`** inside |
| **`import numpy`**, **`import os`** | No arbitrary stdlib | **`from libgeo import ...`** or local **`.r4d`** modules only |
| **`import *`** | Unsupported | **`from mod import a, b`** |
| **`def foo():`** without **`-> type`** | Often still parsed but avoid for clarity | **`def foo() -> None:`** with annotations on public APIs |
| **`cosmos[i] = cosmos[i] * rot`** in **`par for i in range(N)`** | Subscript + par capture differs from **`par for p in list`** pattern | **`par for p in cosmos:`** then **`p = p * rot`** (see demos) |
| **`list[vec4]` size 10_000_000** in examples for CI | May be slow / memory-heavy | Use **`1_000_000`** or **`1024`** unless user asks for scale |
| **`open("file.txt")`**, **`requests.get`** | No builtins | C runtime / future FFI — not here |
| **`async` / `await`** | Not in lowering path | Sequential **`def`** only |
| **`lambda`** | Unsupported surface | Named **`def`** |
| **`try` / `except`** | Limited / not the native error model | Let compile fail; fix types |
| **`# type: ignore`** | Meaningless | Fix the type |
| **`__main__` guard** | N/A — entry is **`main()`** | Top-level **`def main()`** only |
| **`if __name__`** | N/A | Same |
| **`numpy.array`**, **`torch`** | Wrong language | **`list[vec4]`** + **`par`** |

If the user asks for “Python with 4D flavor,” **refuse** and output **valid `.r4d`** from this spec.

---

## 22. Compiler and linker error catalog

Use this table to map **stderr** to **fixes** (also see §17).

| Symptom / message | Likely cause | Fix |
|-------------------|--------------|-----|
| `source file not found` | Wrong **cwd** or typo | Use **absolute path** or `cd` to repo |
| `not a Roma4D source file` | `.py` / `.txt` / wrong ext | Rename to **`.r4d`** |
| `roma4d.toml not found` | File outside package tree | Move under root or set **`R4D_PKG_ROOT`** |
| `ImportError: No module named` | Missing **`libfoo.r4d`** (or legacy suffix) | Create module or fix **`from`** name |
| `import * is not supported` | Python habit | List names explicitly |
| `UseAfterMoveError` / `soa field` | Second read without write-back | Reassign **`cell.pos = ...`** before re-read |
| `BorrowError` / `mutborrow` | Overlapping borrows | Shrink borrow scope; follow §10 |
| `TaintError` | `print` taint into **linear** slot | Do not flow **`print`** into **`soa`** |
| `clang: ... mm_malloc.h` | Clang fallback + MSYS headers | Install **Zig** (default), or **`R4D_GNU_ROOT`** / MinGW on PATH |
| `undefined reference` / link fail | Missing CRT / **`-lm`** | Linux: driver adds **`-lm`**; Windows: MinGW **`bin`** on PATH |
| LLVM `ret` / `void` mismatch | (Fixed in toolchain) **`return None`** in **`-> None`** | Update **`r4`**; explicit void return lowering |
| `synthesized return` | Missing **`return`** in **`-> int`** | Add **`return 0`** or real exit code |
| Parse / `INDENT` / `DEDENT` | Tabs/spaces mix | Reindent consistently |

---

## 23. Ergonomics: `r4`, PATH, `R4D_PKG_ROOT`

- **`r4`** — shortest command; same as **`r4d`**.
- **`.\r4d.ps1`** — developer loop: rebuild **`r4`**, **`r4d`**, **`roma4d`** then run **`r4`**.
- **`go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d`** — install into **`GOPATH/bin`**.
- **`Install-R4dUserEnvironment.ps1`** — user **PATH** + **`R4D_PKG_ROOT`** on Windows.
- Future-proofing: extensions and resolution live in **`source_ext.go`** — add new legacy suffixes there if ever needed.

---

## 24. LLM hard rules (non-negotiable)

1. Output **only** syntactically valid **Roma4D** **`.r4d`** (unless user explicitly requests a legacy extension).
2. **Never** pretend **`print`**, **`time()`**, or **f-strings** behave like CPython.
3. **Never** invent **stdlib** imports beyond **`from <local_module> import ...`**.
4. **Always** use **`vec4` / `rotor` / `multivector`** types for **Cl(4,0)** — never **`Vec4`** / **`float4`**.
5. **Always** end **`-> int`** `main` with an **integer `return`**.
6. If unsure about a feature, **omit** it — prefer **`print("ok")`** + **`pass`** over hallucinated APIs.
7. After code, tell the user to run **`r4d <file>.r4d`** (or **`r4 run`**) and to read **`last_build_failure.log`** on failure.

---

## Document maintenance

When the compiler changes (new builtins, stricter ownership, new keywords), update:

- `src/compiler/typechecker.go` — `seedBuiltins`, import rules.
- `src/compiler/source_ext.go` — **`.r4d` / `.r4s` / `.roma4d`** resolution.
- `src/ai/expert.go` — forgiving **Expert** hints (native rules; no LLM).
- `src/parser/token.go` — keyword set.
- `src/compiler/codegen_llvm.go` — void returns, ABI.
- `src/compiler/llvm_link.go` — **Zig `cc` (Windows)** / **clang** driver for `.ll` + `rt/*.c`.
- `internal/cli/cli.go` — help text, **`ensureSourceFile`**.
- `rt/roma4d_rt.c` — runtime symbols and demos.
- **This guide** — §7, §14–§15, §17–§24.

---

*End of Roma4D Guide.*
