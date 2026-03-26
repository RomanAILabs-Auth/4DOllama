# Roma4D — The World’s First 4D Spacetime Programming Language

**Compile-time spacetime reasoning meets Python-clear syntax and systems-grade native codegen.** Roma4D is a research language where **Cl(4,0) geometry**, **structure-of-arrays data**, and **explicit parallelism** are first-class—not bolted on as libraries.

---

## Key features

- 📖 **Readable surface** — Familiar **Python 3.12–shaped** syntax plus **`par`**, **`soa`**, **`spacetime`**, and related forms.
- 🧮 **Native 4D algebra** — **`vec4`**, **`rotor`**, **`multivector`**, and **`*` / `^` / `|`** as language primitives, not FFI libraries.
- 🔒 **Ownership 2.0** — Linear moves and borrows that match **SoA** columns and **`par`** sendability rules.
- 🌌 **Spacetime regions** — **`spacetime:`** blocks and **`@ t`** for **compile-time** temporal reasoning; lowering stays on the **4D LLVM** path today.
- ⚙️ **Fast native binaries** — **MIR → LLVM IR → clang**; **`-bench`** prints **`load_manifest`** … **`clang_link_exe`** and **`native_run`** (`r4d run` only).
- 🛠️ **Practical toolchain** — **`r4`** / **`r4d`** / **`roma4d`**, **`roma4d.toml`**, and **`debug/last_build_failure.log`** when something breaks.

**Authoritative reference for coding (and for LLM-assisted development):** [`docs/Roma4D_Guide.md`](docs/Roma4D_Guide.md) — syntax, builtins, ownership, spacetime, runtime, and debugging.

---

## Quick start (< 1 minute)

**Prerequisites:** [Go 1.22+](https://go.dev/dl/), **`clang`** on `PATH`, and (on Windows) a **MinGW-w64** toolchain on `PATH` so clang can link with **`-target *-windows-gnu`**—see [Installation](#installation).

```bash
git clone https://github.com/RomanAILabs-Auth/Roma4D.git
cd Roma4D
go build -o "$(go env GOPATH)/bin/r4d" ./cmd/r4d
r4 run examples/min_main.r4s
```

**Windows (recommended):** use the repo launcher so you always run **this** tree’s compiler:

```powershell
cd Roma4D
.\r4d.ps1 run examples\min_main.r4s
```

You should see **`r4d run: passed.`** (exit code **42** is intentional for `min_main`—it returns `42` from `main`).

**Pipeline timings:**

```powershell
.\r4d.ps1 run -bench examples\min_main.r4s
```

---

## Installation

### All platforms

| Step | Command |
|------|---------|
| Clone | `git clone https://github.com/RomanAILabs-Auth/Roma4D.git` and `cd Roma4D`. |
| Build CLI | `go build -o "$(go env GOPATH)/bin/r4d" ./cmd/r4d` — on Windows name the output `r4d.exe` (see `r4d.ps1`). |
| Verify | `r4d version` |

### Windows

1. Install **Go**, **LLVM/Clang** (e.g. official installer), and **MinGW-w64** (e.g. [MSYS2](https://www.msys2.org/) package `mingw-w64-ucrt-x86_64-gcc`).
2. Put **clang** and **MinGW `bin`** on your **user PATH**.
3. From the repo root, run **`.\r4d.ps1 …`** or build with:

   ```powershell
   go build -o "$(Join-Path (go env GOPATH) 'bin\r4d.exe')" ./cmd/r4d
   ```

4. If `r4d` is shadowed by another binary, prepend Go’s bin:  
   `$env:Path = "$(go env GOPATH)\bin;$env:Path"`

5. **Use `r4` from any folder:** run **`.\scripts\Install-R4dUserEnvironment.ps1`** once, then open a new terminal. That adds `%GOPATH%\bin` to your user PATH and sets **`R4D_PKG_ROOT`** so **`r4 run C:\path\to\any.r4s`** works outside the repo. See **`docs/Roma4D_Guide.md`** §3.

**Note:** The driver uses **`-target x86_64-pc-windows-gnu`** (or arm/i686 variants) so you are **not** forced to install Visual Studio’s MSVC libs. Preferring MSVC would require changing the driver in `src/compiler/llvm_link.go`.

### macOS

```bash
brew install go llvm
cd Roma4D
go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d
export PATH="$(go env GOPATH)/bin:$PATH"
r4 run examples/min_main.r4s
```

### Linux

```bash
sudo apt install golang clang   # or your distro’s equivalents
cd Roma4D
go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d
export PATH="$(go env GOPATH)/bin:$PATH"
r4 run examples/min_main.r4s
```

### When builds fail

- Open **`debug/last_build_failure.log`** (clang command, stderr, LLVM IR head).
- Set **`R4D_DEBUG=1`** to mirror the same diagnostics on stderr.

---

## How Roma4D works

Roma4D sits at the intersection of three ideas:

1. **A readable, Python-like surface** so numerical and systems ideas are easy to express and teach.
2. **A systems core**: explicit layout (**SoA**), ownership-friendly field access, and **`par`** regions the compiler can reason about for **SIMD** and future **GPU** backends.
3. **A native 4D spine**: rotors, multivectors, and vectors live in **Cl(4,0)** and lower to **LLVM** with **SIMD-friendly** patterns where the MIR pipeline enables it.

The compiler is implemented in **Go** today (`lexer` → `parser` → **typecheck + Ownership 2.0** → **MIR** → **LLVM IR** → **clang**). Long term, the roadmap includes incremental compilation, richer GPU lowering, and a self-hosted path—see the **ten-pass** list in this repository’s docs and sources.

---

## Language overview

### Python compatibility

Roma4D intentionally **looks like Python** for control flow, functions, indentation, many builtins, and numeric literals—but it is **not** a drop-in Python implementation. Some dynamic features are intentionally restricted so the compiler can emit **fast, predictable native code**.

### New keywords and forms (high level)

| Construct | Role |
|-----------|------|
| **`par`** | Structured parallel loop over ranges / SoA columns; informs sendability and backend hints. |
| **`soa` / `aos`** | Column vs row layout hints on class fields. |
| **`spacetime:`** | Region marker for compile-time temporal/spacetime analysis; **`par`** inside can carry **GPU-par** metadata in MIR. |
| **`unsafe:`** | Explicit low-level region (e.g. raw pointers, manual allocation hooks in MIR). |
| **`vec4`**, **`rotor`**, **`multivector`** | Builtin geometric types. |

(Exact parsing rules live in `src/parser/`; this table is the mental model.)

### Native 4D types

- **`vec4`** — homogeneous 4D vectors (typical use: spatial + projective `w`).
- **`rotor`** — plane + angle style constructors; rotates via the geometric product.
- **`multivector`** — full algebra element; **`^`** (outer) and **`|`** (contraction) disambiguate from integer **XOR** / **OR** where context requires.

### Ownership 2.0

**SoA field access** is modeled as **linear**: read/move out of a slot, use the value, **write back** before reading again—so aliasing and **par** safety stay explicit. Imports and defs carry **Sendable** where needed for parallel regions.

### Spacetime programming

**`t`**, **`expr @ t`**, and **`spacetime:`** blocks participate in **compile-time** staging in the current pipeline: they shape **MIR metadata** and programmer intent without introducing a heavyweight temporal **runtime** on the hot path today.

---

## Core concepts

### SoA by default (mental model)

Think in **columns** for entities: positions, velocities, and attributes are **parallel arrays** or SoA-backed fields, not accidental **array-of-structs** pointer chasing.

```roma4d
class Particle:
    soa pos: vec4
    soa vel: vec4
```

### `par` + SIMD / GPU trajectory

**`par for`** marks a deterministic parallel region. The lowering stack emits **SIMD** patterns for geometric ops (e.g. **`vec4 * rotor`**) where implemented, and records **GPU / spacetime** hints for regions under **`spacetime:`**. Full **CUDA** codegen remains on the roadmap; today you will see **stub** linking when those metadata flags are set—inspect LLVM IR and MIR tests for the exact markers.

```roma4d
spacetime:
    par for p in positions:
        p = p * rot
```

### Spacetime regions and temporal forms

```roma4d
_tau: time = t
sample: vec4 = positions[0] @ t
```

These forms are part of the **language story** for where/when a quantity is evaluated; lowering reuses the same **4D LLVM** paths as non-temporal code in the current implementation.

---

## CLI reference

| Command | Purpose |
|---------|---------|
| **`r4 run <file.r4s> [-bench] [args…]`** | Compile to a temp executable, run it. **`-bench`**: per-phase timings; **`native_run`** is wall time of the child (non-zero exit codes still count as a finished run for bench). |
| **`r4 build <file.r4s> [-o path] [-bench]`** | Emit a persistent executable next to `-o` (default: base name + `.exe` on Windows). |
| **`r4 version`** | Print **`roma4d (r4d) <ver> <os>/<arch>`**. |
| **`r4 help`** | Longer usage text (same as **`r4d`** / **`roma4d help`**). |

**Rules:**

- The source file must live under a directory tree that contains **`roma4d.toml`** (walk upward from the file path).
- **PowerShell:** paste **only** the command line—do not include the **`PS C:\…>`** prompt (on Windows, **`PS`** can alias **`Get-Process`** and break the line).

---

## Repository layout

| Path | Role |
|------|------|
| `roma4d.toml` | Package manifest |
| `src/parser/` | Lexer + parser |
| `src/compiler/` | Typecheck, Ownership 2.0, MIR, LLVM, clang driver, **`-bench`** |
| `src/core/4d/` | Reference Cl(4,0) numerics (tests/tooling) |
| `examples/` | **`.r4s`** samples, **Bench_4d** + Python/Rust baselines |
| `demos/` | **Spacetime Particle Collider** (`spacetime_collider.r4s`) |
| `cmd/r4`, `cmd/r4d`, `cmd/roma4d` | CLI entrypoints |
| `internal/cli` | Shared CLI implementation |
| `r4d.ps1` | Windows helper: `go build` into `GOPATH\bin` + run |

---

## Real-world examples

### Minimal native `main` (`examples/min_main.r4s`)

```roma4d
def main() -> int:
    return 42
```

### Full demo (`examples/hello_4d.r4s`)

The shipped demo exercises **imports**, **SoA** particles, **list comprehensions** over **`vec4`**, **rotor** math, **`spacetime:`** + **`par`**, and an **`unsafe:`** block with MIR allocation helpers.

### Rotor swarm microbench (`examples/Bench_4d.r4d`)

**“Rotor swarm”** style workload over a **`list[vec4]`** with **`par for`** and **`vec4 * rotor`**. Cross-language baselines: **`bench_4d.py`**, **`bench_4d.rs`**, **`run_bench_4d.ps1`**.

### Spacetime Particle Collider (`demos/spacetime_collider.r4s`)

Large-scale demo: **5,000,000** **`vec4`** worldlines, **`spacetime:`** shards (PLAY / PAUSE / **`timetravel_borrow`**), **`par for`** with dual rotors, SoA **`Particle`** beacon, **`unsafe:`** ledger scratch.

```powershell
.\r4d.ps1 run demos\spacetime_collider.r4s
.\r4d.ps1 run -bench demos\spacetime_collider.r4s
```

---

## Spec reference: sample `r4d run -bench` (Collider demo)

Example capture from **`demos/spacetime_collider.r4s`** on a Windows + Clang + MinGW-w64 host (milliseconds vary by machine; sub-ms frontend phases often print as **`0.000`**).

**Pipeline phases (`-bench`):**

```
r4 run -bench <path>/demos/spacetime_collider.r4s
  load_manifest:                  0.000 ms
  read_source:                    0.000 ms
  parse:                          0.531 ms
  typecheck:                      0.000 ms
  ownership_pass:                 0.000 ms
  lower_ast_to_mir:               0.000 ms
  lower_mir_to_llvm:              0.000 ms
  llvm_module_string:             0.000 ms
  write_ll_file:                  0.000 ms
  clang_compile_ll:              58.285 ms
  clang_link_exe:               202.709 ms
  native_run:                   192.540 ms
  total (sum of phases):        454.066 ms
r4d run: passed.
```

**Compiler note:** you may see **`warning: function "main": synthesized return`** — benign for **`main() -> None`** today.

**Program output (ASCII banner; avoids CP437/UTF-8 console mojibake):**

```
  ============================================================
     ROMA4D - SPACETIME PARTICLE COLLIDER (demo build)
  ============================================================

  :: SPACETIME PARTICLE COLLIDER - lattice SIGMA-PRIME
  ------------------------------------------------
  * Chronons in beam        : 5,000,000 (SoA column worldtube)
  * Frames executed         : PLAY  +  PAUSE  +  REWIND(borrow)
  * dt_elapsed (sim ticks)  : 1,048,576 plank quanta (2^20)
  * Temporal collisions     : 1,048,576 (closed light-cone chords)
  * Avg rotor ops / tau     : ~6.29e7 effective (see par region SIMD)
  * Beacon SoA column       : synchronized
  ------------------------------------------------
  >> Collider nominal. Spacetime shards committed to MIR.
  >> For wall-clock and native_run ms: r4 run -bench demos/spacetime_collider.r4s

r4d run: passed (with 1 warning).
```

---

## Performance

| Layer | What to expect |
|-------|----------------|
| **vs CPython** | Hot numeric kernels compiled to **native code** through **LLVM** typically outperform **interpreted** Python loops by a large margin for the same *algorithmic* work—subject to allocator behavior and how much lives in Roma4D vs the host runtime. |
| **vs Rust** | Rust remains the benchmark for **hand-tuned** systems code. Roma4D aims for **safe, explicit layout** and **geometric** productivity first; compare with **`bench_4d.rs`** on your CPU for a concrete scalar loop baseline. |
| **Compile time** | Dominated by **clang** (`clang_compile_ll`, `clang_link_exe` in **`-bench`**). Frontend passes are usually sub-millisecond on small examples. |

Use **`r4d run -bench`** or **`r4d build -bench`** to see **where time goes** in your environment.

---

## License & trademark

- **Roma4D** is released under the **MIT License** — see **[LICENSE](LICENSE)** (Copyright (c) 2026 Daniel Harding - RomanAILabs unless otherwise noted).
- The **Roma4D** name, logos, and related branding are **trademarks** of their respective owners; the MIT license grants rights to the **software**, not to use those marks as your product name without permission. When in doubt, ask the maintainers.

---

## Related: 4DEngine / 4DOllama

Roma4D is also developed inside the **[4DEngine](https://github.com/RomanAILabs-Auth/4DOllama)** monorepo alongside **4DOllama** (Ollama-style API + Rust **`four_d_engine`**). This repository is the **canonical home** for the language toolchain.

---

## Credits

- **Roma4D** — RomanAI Labs / community contributors.
- **4D engine / inference stack** — see **[4DEngine](https://github.com/RomanAILabs-Auth/4DOllama)**.

---

<p align="center"><strong>Roma4D — program in four dimensions; ship native code.</strong></p>
