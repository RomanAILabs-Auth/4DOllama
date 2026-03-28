# RQ4D Master Guide

**RomanAILabs — Daniel Harding**

Single reference for **install**, **daily use**, **programming / extending** the engine, and **LLM-assisted development** of [RomaQuantum4D](https://github.com/RomanAILabs-Auth/RomaQuantum4D) (RQ4D).

---

## Table of contents

1. [Install guide](#1-install-guide)  
2. [User guide](#2-user-guide)  
3. [Programming guide](#3-programming-guide)  
4. [LLM briefing pack](#4-llm-briefing-pack-advanced)  
5. [Quick reference](#5-quick-reference)  
6. [Theoretical foundations (manuscript, deciphered)](#6-theoretical-foundations-manuscript-deciphered)

---

## 1. Install guide

### 1.1 Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Go** | **1.22+** (`go version`). [https://go.dev/dl/](https://go.dev/dl/) |
| **Git** | To clone the repository. |
| **PowerShell** | **5.1+** or **PowerShell 7+** for `scripts/RQ4D_World_Record.ps1` (optional). |

RQ4D is **pure Go** — no CGO, no external quantum SDKs, no Python runtime for the core engine.

### 1.2 Clone

```bash
git clone https://github.com/RomanAILabs-Auth/RomaQuantum4D.git
cd RomaQuantum4D
```

Your folder name may be `RQ4D` or `RomaQuantum4D` depending on how you cloned; the **module root** is the directory that contains `go.mod`.

### 1.3 Verify toolchain

```bash
go version
go build -o rq4d ./cmd/rq4d        # Unix / macOS binary name
go build -o rq4d.exe ./cmd/rq4d    # Windows
go install ./cmd/rq4d              # installs `rq4d` to $GOPATH/bin or $GOBIN
```

On Windows, `rq4d.exe` is gitignored at repo root when named `/rq4d.exe` — local builds are fine.

### 1.4 First run (smoke test)

```bash
go run ./cmd/rq4d examples/manifold_sweep.rq4d
```

You should see **`Executing RQ4D (geometric simulation scale...)`**, eight `MEASURE q[i]` lines, and the **honest telemetry** footer (global passes, bytes touched, **FNV-1a checksum**).

### 1.5 Optional: large-scale geometric demo script

From repo root:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\RQ4D_World_Record.ps1
```

This builds the binary, generates a large `.rq4d` under `examples/`, and runs it. Large `-QubitCount` values create huge scripts; each **CNOT** and each **global pass** is **O(n)** over the full register — start with defaults, then scale up.

### 1.6 Troubleshooting

| Symptom | Likely cause | Action |
|---------|----------------|--------|
| `go: cannot find module` | Not at module root | `cd` to folder containing `go.mod`. |
| Parse error on script | Wrong path or bad opcode | Check spelling; see [§5](#5-quick-reference). |
| Green banner shows as escape codes | Terminal without ANSI | Use Windows Terminal, or ignore raw `\033[32m`. |
| Slow / OOM on huge scripts | **O(n)** global passes per batch + **O(n)** CNOT ripple | Reduce qubit count, use `-GenerateOnly`, or try `--truth-mode` only on small scripts. |

### 1.7 Relation to **Roma4D** (`.r4d`)

- **RQ4D** runs **`.rq4d` text scripts** with the **RQ4D instruction set** (ALLOC, H, X, CNOT, MEASURE).  
- **Roma4D** (`r4d`) compiles **`.r4d` source language** (different syntax: `def`, `vec4`, `rotor`, `par for`, etc.).  
- Files like `examples/spacetime_ui_v3.r4d` in this repo are **Roma4D-style samples** for narrative alignment; execute them with **`r4d`**, not **`rq4d`**.

---

## 2. User guide

### 2.1 Command-line interface

```text
rq4d [flags] <file.rq4d>
```

- **One positional argument**: path to a script file.  
- **`--truth-mode`**: same **ALLOC n** and register size as default mode; disables **H**/**X** parallel batching and runs one **O(n) global pass** after **every** gate line.  
- **No argument** or **missing / unreadable script path**: stderr message, exit **2**.  
- **Invalid script** (parse or run validation error): stderr message, exit **1**.

Execution is sequential by script line; **consecutive `H` or `X`** lines of the **same opcode** may share one parallel apply + **one** global pass (unless `--truth-mode`).

### 2.2 Script format (`.rq4d` for RQ4D engine)

- **One instruction per line** (whitespace-separated tokens).  
- **Case-insensitive** opcodes (`H`, `h`, `H` are equivalent).  
- **Comments**: lines starting with `#` are ignored.  
- **Blank lines** ignored.

### 2.3 Instruction set

| Opcode | Syntax | Meaning |
|--------|--------|---------|
| **ALLOC** | `ALLOC n` | Allocate `n` qubits (`n` ≤ **2²⁴**), each **\|0⟩** (scalar `1`). Followed by an **O(n) global pass** (memory touch + normalization bookkeeping). |
| **H** | `H i` | Hadamard on the **scalar / e₁** computational slice. Consecutive **`H`** lines batch in parallel (worker pool) + **one** global pass, unless **`--truth-mode`**. |
| **X** | `X i` | Pauli **X** via **swap** of blades **0** and **1**. Consecutive **`X`** batch like **H**. |
| **CNOT** | `CNOT c t` | **Field-blended** conditional **X** on target from local **P(\|1⟩)**, control **energy field**, **global phase**, **coherence**, plus a **deterministic spread** term (reproducible; not `math/rand`). Then **O(n) ripple** on all lanes. |
| **MEASURE** | `MEASURE` | Print **P(\|0⟩)**, **P(\|1⟩)** from normalized **scalar / e₁**; updates entropy/coherence; ends with a **global pass**. |

Indices are **0-based**. **`ALLOC` must appear before** gates that use qubits.

### 2.4 Built-in examples (paths relative to repo root)

| File | Intent |
|------|--------|
| `examples/manifold_sweep.rq4d` | 8 qubits, parallel **H**, **MEASURE** — flagship demo. |
| `examples/parallel_h.rq4d` | 4-way parallel **H** batch. |
| `examples/cnot_demo.rq4d` | **X** on control, **CNOT**, both in **\|1⟩** (product state). |
| `examples/spacetime_ui_v3.r4d` | Roma4D **worldtube** demo (run with **`r4d`**, not `rq4d`). |

### 2.5 Output semantics

- Banner: **`Executing RQ4D (geometric simulation scale, Cl(4,0) register)...`**  
- **MEASURE** lines: `MEASURE q[k]: P(|0>)=... P(|1>)=...`  
- Footer **telemetry**: register size, wall time, **total gate ops**, **global pass** count and aggregate time, **average time per global pass**, **bytes touched** (pass accounting), **derived gate ops/s** (TotalOps / wall time), **FNV-1a checksum** (register + globals + **script-order trace** + gate/pass counts).  
- This is a **geometric simulation** on a multivector field; it is **not** a claim of physical hardware qubits.

### 2.6 PowerShell: `scripts/RQ4D_World_Record.ps1`

**Universal paths** — resolves engine root from the script’s location (`scripts/` → parent = repo root).

| Parameter | Role |
|-----------|------|
| `-QubitCount` | Default `65536`; max `131072`. |
| `-EngineRoot` | Override repo root. |
| `-OutScriptName` | Generated file name under `examples/`. |
| `-MirrorDir` | Optional copy of script + `.rq4d` elsewhere. |
| `-GenerateOnly` | Write `.rq4d` only; do not run. |
| `-SkipBuild` | Assume binary already built. |

---

## 3. Programming guide

### 3.1 Repository layout

```text
cmd/rq4d/main.go          # CLI, flags (--truth-mode), parse → quantum.Run, telemetry footer
internal/math/clifford.go # Cl(4,0): Multivector, GeometricProduct, Rotor, Normalize, Reverse
internal/quantum/bridge.go# Global system, gates, O(n) global pass, CNOT ripple, checksum, Run()
internal/parser/lexer.go # .rq4d line parser → []Instruction
examples/*.rq4d           # Sample circuits (and some Roma4D-style .r4d companions)
scripts/*.ps1             # Automation / large manifold generation
go.mod                    # module github.com/RomanAILabs-Auth/RomaQuantum4D
```

### 3.2 Geometric model (non-negotiable design choices)

- **Algebra**: **Cl(4,0)** — Euclidean signature `(+,+,+,+)` on four basis vectors **e₁…e₄**.  
- **Storage**: `Multivector` holds **16** `float64` coefficients; basis blade index = **4-bit bitmask** (bit `i` ⟺ **e_{i+1}** present).  
- **No complex matrices**, no full **2ⁿ** Hilbert-space statevector. Each lane is a **16D multivector**, coupled by **shared global fields** and **O(n) passes** so the simulator does not pretend independent per-qubit scaling is “free.”  
- **Geometric product** remains available in `internal/math` for rotors and extensions; the scripted **H** gate uses the explicit **2×2** action on **C[0], C[1]** for correct superposition in that slice.

### 3.3 Qubit encoding

| State | Multivector |
|-------|-------------|
| **\|0⟩** | Scalar blade **1.0** (index 0), rest 0. |
| **\|1⟩** | **e₁** blade (index 1) normalized usage via gates. |
| **Hadamard** | **(a₀,a₁) → ((a₀+a₁)/√2, (a₀−a₁)/√2)** on blades **0** and **1**. |
| **X (script / CNOT)** | **Swap** blades **0** and **1**. |

### 3.4 CNOT semantics (current engine)

- **Target update** is **not** `P(|1⟩) > ½` alone. The engine blends **local P(|1⟩)**, the control’s **energy field** entry, **global phase** (sinusoidal window), **coherence**, and a **deterministic** pseudo-spread derived from **TraceHash** + control multivector + indices (no `math/rand`).  
- **Non-local ripple**: every CNOT performs an **O(n)** update over **all** lanes (distance-weighted coupling plus a small uniform tail) and updates **global** entropy. This remains a **geometric simulation**, not a full **2ⁿ** statevector.

### 3.5 Parallelism and global passes

- **ALLOC**: one **O(n)** **global pass** immediately after allocation (touches every lane).  
- **Runs of `H` or `X`**: worker-pool parallel apply over the batch indices, then **one** **O(n)** **global pass** (unless **`--truth-mode`**, which runs **pass per line**).  
- **CNOT / MEASURE**: each ends with a **global pass**.  
- **Checksum**: **FNV-1a** over **all** floats in the register + **GlobalSystem** — intended to change when work runs and state changes.

### 3.6 Extending the engine

1. **New opcode**  
   - Add `Op*` constant and `parseLine` branch in `internal/parser/lexer.go`.  
   - Extend `Instruction` fields if needed.  
   - Handle in `internal/quantum/bridge.go` **`Run`** (respect **`--truth-mode`** and schedule **global passes** honestly).

2. **New gate**  
   - Implement in `internal/quantum/bridge.go` using **`gamath.GeometricProduct`** / **`Rotor`**.  
   - Keep **`github.com/RomanAILabs-Auth/RomaQuantum4D/internal/math`** import alias **`gamath`** to avoid clashing with **`math`**.

3. **Joint quantum state**  
   - Would require a **new representation** (e.g. dedicated tensor or GA in higher dimension), **not** a drop-in patch to the current per-qubit `[]Multivector`.

### 3.7 Copyright / file headers

Go sources in this project use:

```text
// <filename>.go
// Copyright RomanAILabs - Daniel Harding
```

Preserve this when adding files.

---

## 4. LLM briefing pack (advanced)

Use this section as **system or user context** when asking an LLM to modify RQ4D.

### 4.1 One-paragraph project truth

> **RomaQuantum4D (RQ4D)** is a **Go 1.22** module **`github.com/RomanAILabs-Auth/RomaQuantum4D`** that runs **`.rq4d` scripts** (**ALLOC, H, X, CNOT, MEASURE**) as a **geometric simulation** on **Cl(4,0)** lanes (16 `float64`s each). A **global system** (phase, per-lane energy field, coherence) and **mandatory O(n) global passes** couple the register; **CNOT** adds an **O(n) ripple** and a **field-blended** target update. **`--truth-mode`** disables parallel **H**/**X** batching and runs a global pass **per gate line**. The **checksum** includes register data, globals, **script-order trace**, and **gate/pass counts** — traceability only, not a hardware claim.

### 4.2 Hard rules for generated changes

1. **Do not** replace the GA core with `complex128` matrices or NumPy-style statevectors unless the user explicitly requests a **new subsystem** and accepts a **breaking redesign**.  
2. **Do not** import standard **`math`** as **`math`** in `internal/quantum` if it shadows **`github.com/RomanAILabs-Auth/RomaQuantum4D/internal/math`** — use **`gamath`** / **`stdmath`**.  
3. **Preserve** honest telemetry (**global passes**, **checksum**, **geometric simulation scale** wording) unless the user requests a deliberate format change.  
4. **Parser** must remain **line-based**, **fail with clear errors** (file:line), and **ignore `#`**.  
5. **Batch parallel `H` / `X`** for consecutive lines when **`--truth-mode` is off**; honor **`--truth-mode`** by **not** batching.  
6. **`.r4d` Roma4D language** files in `examples/` are **not** parsed by `rq4d`; do not assume they load in the Go engine.

### 4.3 File map (where to edit what)

| Task | Primary files |
|------|----------------|
| New script opcode | `internal/parser/lexer.go`, `internal/quantum/bridge.go` (`Run`) |
| Gate / measure / global pass | `internal/quantum/bridge.go` |
| GA product / rotor | `internal/math/clifford.go` |
| UX / banner / flags | `cmd/rq4d/main.go` |
| Large demo generation | `scripts/RQ4D_World_Record.ps1` |
| User-facing docs | `README.md`, **`docs/RQ4D_MASTER_GUIDE.md`** |

### 4.4 Suggested prompts (copy-paste)

**Refactor**

> In `github.com/RomanAILabs-Auth/RomaQuantum4D`, extract instruction execution from `main` into a dedicated `internal` package without changing observable output or the `.rq4d` ISA.

**Opcode**

> Add opcode `Y [target]` that applies a documented Cl(4,0) operator (left multiply) to qubit `target`, batch consecutive `Y` lines like `H`, and update `docs/RQ4D_MASTER_GUIDE.md`.

**Performance**

> Reduce goroutine fan-out for large `H` blocks by processing in chunks of 1024 with a worker pool; keep results identical to the current engine within float tolerance.

**Tests**

> Add `internal/math` table-driven tests for `GeometricProduct` on basis blades and golden values for `e12*e12 == -1` in the scalar blade.

### 4.5 Hallucination guardrails

- There is **no** built-in GPU, **no** Qiskit, **no** automatic Roma4D compiler bridge in this repo.  
- **Repository URL**: **https://github.com/RomanAILabs-Auth/RomaQuantum4D**  
- Module path: **`github.com/RomanAILabs-Auth/RomaQuantum4D`**

---

## 5. Quick reference

### 5.1 ISA (RQ4D `.rq4d` engine)

```text
ALLOC n
H i
X i
CNOT c t
MEASURE
```

### 5.2 Commands

```bash
go run ./cmd/rq4d examples/manifold_sweep.rq4d
go build -o rq4d ./cmd/rq4d && ./rq4d examples/cnot_demo.rq4d
```

```powershell
pwsh -File .\scripts\RQ4D_World_Record.ps1 -QubitCount 8192
```

### 5.3 Links

- **GitHub**: [RomanAILabs-Auth/RomaQuantum4D](https://github.com/RomanAILabs-Auth/RomaQuantum4D)  
- **Go**: [https://go.dev/doc/](https://go.dev/doc/)

---

## 6. Theoretical foundations (manuscript, deciphered)

This section **restates** the RomanAILabs “Spacetime Engine / Q-tensor” manuscript and the related **PowerShell + Python** snippets in **clear engineering language**, and explains **how it relates to RQ4D** versus **Roma4D**.

### 6.1 Deciphering the snippets (ground truth about the artifacts)

**What you are looking at**

1. **Outer PowerShell** (`$PSCode` → `Deploy_Roma4D.ps1` / `Deploy_Roma4D_Quantum.ps1`): creates `...\RomanAILabs\4d_Spacetime_engine`, drops a **`.py`** file, runs **`python`**.  
2. **`Roma4D_Tesseract.py`**: real-time **ASCII** “render” of a **4D hypercube** using **stereographic projection** \(\mathbb{R}^4 \to \mathbb{R}^3\) and a simple **XW** rotation, then a **3D→2D** screen projection.  
3. **`Roma4D_Hyper_Quantum_Core.py`**: **NumPy** demo that (a) applies **two commuting plane rotations** (XY and ZW) with the same angle as an **isoclinic-style** coordinate demo, and (b) uses **random** small matrices as a **stand-in** for Qwen-style \(Q,K\) to produce a **Frobenius-norm** scalar that bumps a 4D lattice cell.

**What they are not**

- They are **not** the **Go** `rq4d` / `rq4d-core` binaries.  
- They are **not** a full **Cl(4,0) multivector engine** in Python (no general geometric product / 16-blade simulation in those snippets).  
- They are **not**, by themselves, a live **Ollama** integration—only a **toy** unless you connect real inference APIs and define stable tensor→field maps.

### 6.2 Mathematical core (professional summary)

| Topic | Statement |
|-------|-----------|
| **Algebra** | Vectors in **\(Cl(4,0)\)** as \(P = x e_1 + y e_2 + z e_3 + w e_4\) with \(e_i^2=1\), \(e_ie_j=-e_je_i\) (\(i\neq j\)). |
| **Rotors** | Rotations via **even elements** \(R=\exp(-\tfrac{\theta}{2}B)\) with bivector plane \(B\); **sandwich** \(P'=RP\tilde R\) implements **SO(4)**-type motion without ad-hoc Euler chains in 4D. |
| **\(w\) vs time** | Manuscript treats **\(w\)** as a **fourth spatial** coordinate; **engine time** is a **separate** tick—useful as a **4+1 simulation** convention. |
| **4D waves** | Discrete **Laplacian** on a **4D grid** + second-order time stepping for \(\phi\): standard explicit PDE pattern for ripples / interference stories. |
| **Isoclinic rotations** | In **SO(4)**, a **double rotation** in two orthogonal planes can use the **same angle** (isoclinic case). The Python demo implements the **commuting split** as **two 2D rotations** on coordinates—consistent as a **model**, not the full multivector derivation. |
| **Hodge / dual planes** | In **4D Euclidean** exterior algebra, **2-planes** pair with orthogonal **2-planes**; **\(Cl(4,0)\)** provides a disciplined language for that **duality** when you fix conventions. |
| **Q-tensor coupling** | **Forward:** map field features (e.g. **\(w\)-axis** gradients) into scalars that **modulate** a downstream computation. **Reverse:** map norms (e.g. attention-energy proxies) into **source terms** for \(\phi\)—requires a **defined API** and stability analysis for any nonlinear PDE claim. |

### 6.3 “Bidirectional spacetime–cognition coupling” (vision, not a warranty)

The manuscript’s **cognitive gravity**, **Hodge-dual attention mapping**, and **multi-model wells** are **design narratives**: they describe **what could be built** when:

- inference tensors are **observable** through a **bounded, user-space** bridge (e.g. HTTP to a local server), and  
- field updates remain **deterministic or traceable** per your product’s telemetry rules.

They are **not** automatic properties of every RomanAILabs checkout unless that integration is **implemented and tested**.

### 6.4 How this applies to **RQ4D** (this repository)

| Layer | Role |
|-------|------|
| **Manuscript** | **North-star physics / GA story** for RomanAILabs: **4D spatial** thinking, **rotors**, **waves**, optional **AI coupling**. |
| **RQ4D Go tree** | **Shipping** concerns: **lattice / tensor-style** numerics, **CLI**, **RQ4D-CORE** daemon, **HTTP bridge** patterns, **DAG / executor** modules—each must be documented in its own build’s **README** and code. |
| **Alignment work** | Connecting manuscript operators to code means: **explicit kernels**, **macro-node** schedulers, **documented** I/O to Ollama (or others), and **honest** labels (simulation scale, not hardware). |

**LLM guardrail:** when the user pastes the **PowerShell/Python** bundle, explain that it is a **demo launcher**, then point to **this §6** and to the **actual** entrypoints (`go run ./cmd/rq4d`, `rq4d core`, etc.) for the **Go** product.

### 6.5 Speculative “horizons” (framed honestly)

Applications such as **4D “hidden” cryptography**, **inference time dilation via lattice density**, or **protein folding via \(w\) rotation** are **hypotheses**. List them as **research directions**, cite required **evidence** (benchmarks, invariants, safety), and do not present them as finished **RQ4D** features unless the repository **ships** them behind tests.

---

*End of RQ4D Master Guide — RomanAILabs / Daniel Harding.*
