# Roma4D — The World’s First 4D Spacetime Programming Language

**Compile-time spacetime reasoning meets Python-clear syntax and systems-grade native codegen.** Roma4D is a research language where **Cl(4,0) geometry**, **structure-of-arrays data**, and **explicit parallelism** are first-class—not bolted on as libraries.

---

## Key features

- 📖 **Readable surface** — Familiar **Python 3.12–shaped** syntax plus **`par`**, **`soa`**, **`spacetime`**, and related forms.
- 🧮 **Native 4D algebra** — **`vec4`**, **`rotor`**, **`multivector`**, and **`*` / `^` / `|`** as language primitives, not FFI libraries.
- 🔒 **Ownership 2.0** — Linear moves and borrows that match **SoA** columns and **`par`** sendability rules.
- 🌌 **Spacetime regions** — **`spacetime:`** blocks and **`@ t`** for **compile-time** temporal reasoning; lowering stays on the **4D LLVM** path today.
- ⚙️ **Fast native binaries** — **MIR → LLVM IR** then **Windows: `zig cc`** (default) or **clang**; **Unix: clang**; **`-bench`** prints **`zig_*` / `clang_*`** and **`native_run`** (`r4d run` only).
- 🛠️ **Practical toolchain** — **`r4d`** / **`roma4d`**, **`roma4d.toml`**, and **`debug/last_build_failure.log`** when something breaks.

**What you can build (vision + applications):** [`roma4d/docs/User_Guide.md`](roma4d/docs/User_Guide.md).  
**Full programming reference (share with an LLM):** [`roma4d/docs/Roma4D_Guide.md`](roma4d/docs/Roma4D_Guide.md).

**One-file manual:** **[roma4d/docs/Roma4D_Master_Guide.md](roma4d/docs/Roma4D_Master_Guide.md)** · **Doc hub:** **[roma4d/docs/README.md](roma4d/docs/README.md)** · **Install:** **[roma4d/docs/Install_Guide.md](roma4d/docs/Install_Guide.md)** · Root stub: [INSTALL.md](INSTALL.md).

---

## Quick start (< 1 minute)

**Prerequisites:** [Go 1.22+](https://go.dev/dl/). **Windows:** [**Zig**](https://ziglang.org/download/) on `PATH` (default: `zig cc`; optional **`R4D_ZIG`** for a full path to `zig.exe`). **Unix/macOS:** **`clang`** on `PATH`. **Windows fallback** if Zig is missing: LLVM **clang** + **MinGW-w64**—see [Installation](#installation).

```bash
cd roma4d
go build -o "$(go env GOPATH)/bin/r4d" ./cmd/r4d
r4d examples/min_main.r4d
```

**Windows (recommended):** use the repo launcher so you always run **this** tree’s compiler:

```powershell
cd roma4d
.\r4d.ps1 examples\min_main.r4d
```

You should see **`r4 run: passed.`** (exit code **42** is intentional for `min_main`—it returns `42` from `main`).

**Pipeline timings:**

```powershell
.\r4d.ps1 run -bench examples\min_main.r4d
```

---

## Installation

### All platforms

| Step | Command |
|------|---------|
| Clone | `git clone` this repository and `cd` into it. |
| Enter module | `cd roma4d` |
| Build CLI | `go build -o "$(go env GOPATH)/bin/r4d" ./cmd/r4d` — on Windows name the output `r4d.exe` (see `r4d.ps1`). |
| Verify | `r4d version` |

### Windows

1. Install **Go** and [**Zig**](https://ziglang.org/download/) on **PATH** (recommended on Windows).
2. **Fallback:** LLVM **Clang** + **MinGW-w64** (e.g. MSYS2) if not using Zig.
3. From `roma4d/`, run **`.\r4d.ps1 …`** or build with:

   ```powershell
   go build -o "$(Join-Path (go env GOPATH) 'bin\r4d.exe')" ./cmd/r4d
   ```

4. If `r4d` is shadowed by another binary, prepend Go’s bin:  
   `$env:Path = "$(go env GOPATH)\bin;$env:Path"`

**Note:** The driver uses **`-target *-windows-gnu`** (Zig or clang) so you are **not** forced to install Visual Studio’s MSVC libs. Preferring MSVC would require changing the link driver (`src/compiler/llvm_link.go`, Zig argv in `zig_link.go`).

### macOS

```bash
brew install go llvm
cd roma4d
go install ./cmd/r4d ./cmd/roma4d
export PATH="$(go env GOPATH)/bin:$PATH"
r4d examples/min_main.r4d
```

### Linux

```bash
sudo apt install golang clang   # or your distro’s equivalents
cd roma4d
go install ./cmd/r4d ./cmd/roma4d
export PATH="$(go env GOPATH)/bin:$PATH"
r4d examples/min_main.r4d
```

### When builds fail

- Open **`roma4d/debug/last_build_failure.log`** (zig/clang command, stderr, LLVM IR head).
- Set **`R4D_DEBUG=1`** to mirror the same diagnostics on stderr.

---

## How Roma4D works

Roma4D sits at the intersection of three ideas:

1. **A readable, Python-like surface** so numerical and systems ideas are easy to express and teach.
2. **A systems core**: explicit layout (**SoA**), ownership-friendly field access, and **`par`** regions the compiler can reason about for **SIMD** and future **GPU** backends.
3. **A native 4D spine**: rotors, multivectors, and vectors live in **Cl(4,0)** and lower to **LLVM** with **SIMD-friendly** patterns where the MIR pipeline enables it.

The compiler is implemented in **Go** today (`lexer` → `parser` → **typecheck + Ownership 2.0** → **MIR** → **LLVM IR** → **Zig `cc` (Windows)** / **`clang`**). Long term, the roadmap includes incremental compilation, richer GPU lowering, and a self-hosted path—see **`roma4d/README.md`** (short pointer) and the **ten-pass** list in the source tree.

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
| **`r4d <file.r4d> [args…]`** / **`r4 run [--strict] <file.r4d> [-bench] [args…]`** | Temp build + run. Default: forgiving **Expert** hints on failure; **`--strict`**: raw errors only. |
| **`r4 build [--strict] <file.r4d> [-o path] [-bench]`** | Emit a persistent executable next to `-o` (default: base name + `.exe` on Windows). |
| **`r4d version`** | Print **`roma4d (r4d) <ver> <os>/<arch>`**. |
| **`r4d help`** | Longer usage text (same as **`roma4d help`**). |

**Rules:**

- The source file must live under a directory tree that contains **`roma4d.toml`** (walk upward from the file path).
- **PowerShell:** paste **only** the command line—do not include the **`PS C:\…>`** prompt (on Windows, **`PS`** can alias **`Get-Process`** and break the line).

---

## Real-world examples

### Minimal native `main` (`examples/min_main.r4d`)

```roma4d
def main() -> int:
    return 42
```

### Full demo (`examples/hello_4d.r4d`)

The shipped demo exercises **imports**, **SoA** particles, **list comprehensions** over **`vec4`**, **rotor** math, **`spacetime:`** + **`par`**, and an **`unsafe:`** block with MIR allocation helpers—see the file under **`roma4d/examples/`**.

### Rotor swarm microbench (`examples/Bench_4d.r4d`)

**“Rotor swarm”** style workload over a **`list[vec4]`** with **`par for`** and **`vec4 * rotor`**:

```roma4d
def main() -> None:
    n: int = 200_000
    rot: rotor = rotor(angle=1.5707963267948966, plane="xy")
    pos: list[vec4] = [vec4(x=0, y=0, z=0, w=1) for _ in range(n)]
    par for p in pos:
        p = p * rot
    print("Bench_4d roma4d: done (see bench_4d.py / bench_4d.rs for timed scalar loops)")
```

**Cross-language baselines** (same folder): **`bench_4d.py`**, **`bench_4d.rs`**, and **`run_bench_4d.ps1`** document how to compare **interpreted Python**, **rustc -O**, and **`r4d run`** on your machine. Read the header comments in **`Bench_4d.r4d`**—today’s MIR lowering does not model every Python **`for`** as a tight scalar inner loop, so numbers are **illustrative of the 4D lane**, not apples-to-apples loop parity.

### Spacetime Particle Collider (`demos/spacetime_collider.r4d`)

Large-scale demo: **5,000,000** **`vec4`** worldlines, **`spacetime:`** shards (PLAY / PAUSE / **`timetravel_borrow`**), **`par for`** with dual rotors, SoA **`Particle`** beacon, **`unsafe:`** ledger scratch.

```powershell
cd roma4d
.\r4d.ps1 demos\spacetime_collider.r4d
.\r4d.ps1 run -bench demos\spacetime_collider.r4d
```

---

## Spec reference: sample `r4d run -bench` (Collider demo)

Example capture from **`demos/spacetime_collider.r4d`** on Windows with **Zig** or **Clang+MinGW** (milliseconds vary; sub-ms frontend phases often print as **`0.000`**).

**Pipeline phases (`-bench`):**

```
r4 run -bench <path>/demos/spacetime_collider.r4d
  load_manifest:                  0.000 ms
  read_source:                    0.000 ms
  parse:                          0.531 ms
  typecheck:                      0.000 ms
  ownership_pass:                 0.000 ms
  lower_ast_to_mir:               0.000 ms
  lower_mir_to_llvm:              0.000 ms
  llvm_module_string:             0.000 ms
  write_ll_file:                  0.000 ms
  zig_compile_ll / clang_compile_ll:  …
  zig_link_exe / clang_link_exe:      …
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
  >> For wall-clock and native_run ms: r4 run -bench demos/spacetime_collider.r4d

r4d run: passed (with 1 warning).
```

---

## Performance

| Layer | What to expect |
|-------|----------------|
| **vs CPython** | Hot numeric kernels compiled to **native code** through **LLVM** typically outperform **interpreted** Python loops by a large margin for the same *algorithmic* work—subject to allocator behavior and how much lives in Roma4D vs the host runtime. |
| **vs Rust** | Rust remains the benchmark for **hand-tuned** systems code. Roma4D aims for **safe, explicit layout** and **geometric** productivity first; compare with **`bench_4d.rs`** on your CPU for a concrete scalar loop baseline. |
| **Compile time** | Dominated by **`zig cc`** or **clang** in **`-bench`**. Frontend passes are usually sub-millisecond on small examples. |

Use **`r4d run -bench`** or **`r4d build -bench`** to see **where time goes** in your environment.

---

## License & trademark

- **Roma4D** sources under **`roma4d/`** are released under the **MIT License** (see **`roma4d/LICENSE`**; copyright notice: **Daniel Harding – RomanAILabs** unless the file states otherwise).
- The **Roma4D** name, logos, and related branding are **trademarks** of their respective owners; the MIT license grants rights to the **software**, not to use those marks as your product name without permission. When in doubt, ask the maintainers.

---

## Also in this repository: 4DOllama

**4DEngine** is a **monorepo**. **4DOllama** is a separate product: an **Ollama-compatible** HTTP API and **`4dollama`** CLI (Go) with **streaming** NDJSON chat/generate, default bind **127.0.0.1:13377**, and native inference through the Rust **`four_d_engine`** (CGO). The **authoritative product overview**—architecture, languages (Go / Rust / optional Python), streaming behavior, ports, and quick start—is **[`4DOllama/README.md`](4DOllama/README.md)**. Install details, API tables, and Docker: **[`docs/4DOllama.md`](docs/4DOllama.md)**.

---

## Credits

- **Roma4D** — RomanAI Labs / community contributors (see `roma4d/` and git history).
- **4DOllama / 4D engine** — see **`docs/4DOllama.md`** and **`docs/ARCHITECTURE.md`**.

---

<p align="center"><strong>Roma4D — program in four dimensions; ship native code.</strong></p>
