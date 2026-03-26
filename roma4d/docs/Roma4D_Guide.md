# Roma4D Programming Guide

**Purpose:** This document is a **complete, tool-accurate reference** for writing **Roma4D** (`.roma4d`) programs. It is intended for **human developers** and for **feeding to an LLM** so generated code matches the real compiler, typechecker, ownership rules, and native runtime.

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

- **Inputs:** One `.roma4d` source file per build/run, plus any imported `.roma4d` modules under the **package root** (directory containing `roma4d.toml`).
- **Pipeline:** `lexer → parser → typecheck → Ownership 2.0 pass → MIR → LLVM IR → clang` (compile + link).
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
  libgeo.roma4d        # example library module
  examples/*.roma4d
  demos/*.roma4d
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

**Binaries:** `r4d` and `roma4d` share the same implementation (`internal/cli`).

| Command | Meaning |
|--------|---------|
| `r4d help` | Usage text. |
| `r4d version` | Prints `roma4d (r4d) <ver> <os>/<arch>`. |
| `r4d build <file.roma4d> [-o path] [-bench]` | Compile to a persistent executable. Default output name: basename + `.exe` on Windows. |
| `r4d run <file.roma4d> [-bench] [args...]` | Build in a temp dir, run the executable; extra args go to the native `main`. |

**`-bench`:** Emits phase timings (`load_manifest`, `parse`, `typecheck`, `lower_*`, `clang_*`, and `native_run` for `run`).

**PowerShell note (Windows):** When pasting commands, **do not** paste the `PS C:\...>` prompt. On some systems `PS` is an alias and breaks the line.

**Recommended Windows workflow:** From `roma4d/`, use `.\r4d.ps1` — it rebuilds `r4d.exe` into `GOPATH\bin` and prepends PATH so you always use **this** tree’s compiler.

### Run `r4d` from any directory (Windows)

1. **One-time:** From the `roma4d/` repo root, run:

   ```powershell
   .\scripts\Install-R4dUserEnvironment.ps1
   ```

   This **`go install`s** `r4d` / `roma4d` into **`%GOPATH%\bin`**, appends that folder to your **user** `PATH`, and sets **`R4D_PKG_ROOT`** to this repo root (the directory that contains `roma4d.toml`).

2. **Open a new PowerShell window** (so PATH and env vars reload).

3. **Run a file with an absolute path** from anywhere:

   ```powershell
   r4d run C:\Users\You\Desktop\4DEngine\roma4d\demos\cosmic_genesis.roma4d
   ```

   Or keep sketches under `Desktop\` and still compile them:

   ```powershell
   r4d run C:\Users\You\Desktop\my_sketch.roma4d
   ```

   Imports such as `from libgeo import ...` resolve under **`R4D_PKG_ROOT`**, not under the sketch’s folder.

**Environment variables (any OS):**

| Variable | Purpose |
|----------|---------|
| `R4D_PKG_ROOT` | Fallback package root if `roma4d.toml` is not found above the `.roma4d` file path. |
| `ROMA4D_HOME` | Same as `R4D_PKG_ROOT` (either may be set). |

**Relative paths:** `r4d run demos\foo.roma4d` is resolved from your **current working directory**. If you are in `Desktop` and `demos\` does not exist there, create it or pass a **full path** to the file.

---

## 4. Modules and imports

### 4.1 Module file resolution

Given `from foo.bar import x` or `import foo.bar`, the checker resolves **`foo.bar`** to a file under **package root**:

1. `pkgRoot/foo/bar.roma4d` (slashes from dots)
2. `pkgRoot/foo.bar.roma4d` (single segment with dots — less common)

If neither exists → **ImportError**.

### 4.2 `import` forms

- **`import mod`** — binds module name `mod` (or alias with `as`).
- **`from mod import a, b`** — imports exported names from module `mod`.
- **`import *`** — **not supported** (hard error in the typechecker).

### 4.3 What a module exports (current compiler)

When loading a submodule, the compiler collects **top-level** `def` and `class` names into `Exports`. It does **not** fully type-check the submodule body in the same pass as a full second compilation unit (v0 behavior); still, **only names present as top-level defs/classes** are importable.

**Practical rule:** Put shared functions and classes at **module top level** in `libfoo.roma4d`.

### 4.4 Example

`libgeo.roma4d` at package root:

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

**Important:** There is **no** general dynamic string runtime for building HTTP bodies inside `.roma4d` alone; demos that need JSON use **C + curl** in `roma4d_rt.c`.

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

Pattern (from `examples/hello_4d.roma4d`):

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
- **`bump`**, **`identity_v4`** — if you add matching defs, prefer **Roma4D `lib*.roma4d`** for source-level libraries; C duplicates are for ABI demos only.
- **`identity_v4`**, geometric stubs as needed for lowering
- **`ollama_demo`**, **`quantum_server_demo`** — shell out to **`curl`** (see §15)

**Float formatting:** Native `print` is **string-oriented**; numeric `printf` from Roma4d is not the general story yet — demos use **fixed strings** or C-side output for detailed numbers.

---

## 15. Ollama / HTTP demos (builtins)

### 15.1 `ollama_demo()`

- **Requires:** `curl` on `PATH`, local **Ollama** (`ollama serve`, default `http://127.0.0.1:11434`), model pulled (e.g. `qwen2.5`).
- **Behavior:** Fixed JSON body embedded in C (because dynamic host strings are not generally available in `.roma4d` yet).
- **See:** `demos/causal_oracle.roma4d`.

### 15.2 `quantum_server_demo()`

- **Requires:** Same as above.
- **Environment variables:**
  - **`QUANTUM_QUERY`** — optional user question (sanitized for JSON).
  - **`QUANTUM_CONTINUE=1`** — load/save **4-qubit statevector** snapshot file under `TEMP` / `TMPDIR` for **cross-run** continuity.
- **See:** `demos/quantum_server.roma4d`.

**Security note:** These call **`system("curl ...")`** from C — only as trusted as your local shell and paths.

---

## 16. Compilation pipeline (mental model)

1. **Parse** `.roma4d` → AST (`src/parser`).
2. **Typecheck** → types on nodes; imports resolved; builtins known (`src/compiler/typechecker.go`).
3. **Ownership 2.0** → linear SoA + borrow conflicts (`src/compiler/ownership.go`).
4. **Lower to MIR** (`src/compiler/ast_to_mir.go`).
5. **LLVM IR** (`src/compiler/codegen_llvm.go` + helpers).
6. **Clang:**
   - Compile `.ll` → `.o`
   - Link `.o` + `roma4d_rt.c` (+ optional CUDA stub) → executable  
7. **Windows:** `-target x86_64-pc-windows-gnu` (or arm/i686 variants) → **MinGW** linking, not MSVC by default.
8. **Unix:** linker adds **`-lm`** for math in `roma4d_rt.c`.

---

## 17. Help, debugging, and common failures

### 17.1 Build failure log (always check this first)

On failure, the driver appends a structured record to:

**`pkgRoot/debug/last_build_failure.log`**

It includes **stage** (`clang_compile`, `clang_link`, etc.), **full clang argv**, **stderr**, and a **head slice of LLVM IR** for context.

### 17.2 `R4D_DEBUG`

```bash
export R4D_DEBUG=1    # Unix
set R4D_DEBUG=1       # Windows cmd
$env:R4D_DEBUG="1"    # PowerShell
```

Mirrors the same failure block to **stderr** immediately.

### 17.3 “`roma4d.toml` not found”

- Pass a **full path** to a `.roma4d` file that lives **under** a directory tree containing `roma4d.toml`, **or**
- Set **`R4D_PKG_ROOT`** (or **`ROMA4D_HOME`**) to the folder that **contains** `roma4d.toml`, then run `r4d run C:\anywhere\sketch.roma4d`.
- **Windows:** Run **`.\scripts\Install-R4dUserEnvironment.ps1`** once (see §3) to set user **`R4D_PKG_ROOT`** and put **`r4d`** on PATH.

### 17.4 Clang not found

- Install **LLVM**; ensure **`clang`** on `PATH`.
- The finder tries `clang`, `clang-18`, … `clang.exe`.

### 17.5 Windows: MinGW / `mm_malloc.h` / link errors

**Symptom:** Clang includes **`msys64/ucrt64/include/stdlib.h`** but fails on **`mm_malloc.h`**.

**Fix (automatic in recent drivers):** The linker adds **`--gcc-toolchain=...`** when it detects a MinGW root (via `gcc` on `PATH` or common MSYS2 install paths).

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

Add explicit **`return`** in `main` (or all branches) to silence and to match intent.

### 17.9 Parser / indentation errors

Roma4D uses **indentation-based** blocks like Python. **Tabs vs spaces:** stay consistent; mixed indentation can confuse the lexer structure.

---

## 18. LLM checklist (generate valid Roma4D)

When asked to write Roma4D, follow this checklist:

1. **File extension:** `.roma4d`.
2. **Package root:** Assume `roma4d.toml` exists **above** the file; use **`from libgeo import ...`** only if `libgeo.roma4d` exists at root.
3. **Imports:** **`from mod import a, b`** — **never** `import *`.
4. **`main`:** `def main() -> None:` (or `-> int:`) with explicit `return` where needed.
5. **4D math:** Use **`vec4` / `rotor` / `multivector`** and **typed** expressions so `* ^ |` resolve to **algebra**, not bitwise, when intended.
6. **SoA classes:** For `soa` fields, always **read → write back** before second read.
7. **Parallelism:** Put **`par for`** inside **`spacetime:`** for the canonical pattern from demos.
8. **Unsafe:** Only inside **`unsafe:`**; use **`mir_*`** for raw memory games.
9. **I/O:** **`print("literal string")`** — do not assume `print(x)` for arbitrary numeric formatting.
10. **External HTTP / LLM:** Call **`ollama_demo()`** or **`quantum_server_demo()`** from **`unsafe:`** (or assign result) **only** if the user accepts **curl + local Ollama**; mention **env vars** for quantum queries.
11. **After writing:** Recommend **`r4d run`** (or **`.\r4d.ps1 run`** on Windows) and, if it fails, **`debug/last_build_failure.log`**.

**Safe boilerplate skeleton:**

```roma4d
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
```

---

## 19. Example programs in this repo

| Path | Role |
|------|------|
| `examples/min_main.roma4d` | Smallest native `main` → `int` (**exit 42**). |
| `examples/hello_4d.roma4d` | Full **4D + SoA + spacetime + par + unsafe** tour. |
| `demos/spacetime_collider.roma4d` | Large **worldtube** + narrative **frames**. |
| `demos/causal_oracle.roma4d` | **`ollama_demo()`** + spacetime story. |
| `demos/quantum_server.roma4d` | **`quantum_server_demo()`** + **QUANTUM_*** env vars. |
| `libgeo.roma4d` | Tiny **importable** library (`bump`, `identity_v4`). |

---

## Document maintenance

When the compiler changes (new builtins, stricter ownership, new keywords), update:

- `src/compiler/typechecker.go` — `seedBuiltins`, import rules.
- `src/parser/token.go` — keyword set.
- `rt/roma4d_rt.c` — runtime symbols and demos.
- **This guide** — keep §7, §14–§15, and §17 aligned with reality.

---

*End of Roma4D Guide.*
