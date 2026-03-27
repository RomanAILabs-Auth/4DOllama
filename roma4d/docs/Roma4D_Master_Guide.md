# Roma4D Master Guide

### *The complete journey — from empty disk to shipped binary, with Clifford algebra in between*

---

**One book. Every essential guide.** Install the toolchain, learn what 4D programming unlocks, wire up LLMs safely, debug like a pro, read failures, ship binaries — then stay for the **full normative specification** of the language (Part IX).

| | |
| --- | --- |
| **Who this is for** | Developers, researchers, technical artists, and LLM operators working with `.r4d` sources in the **4DEngine** monorepo. |
| **How to use this file** | Read **Parts I → VIII** in order the first time; keep **Part IX** as your permanent law book. Jump via the **table of contents** below. Individual chapters also live as separate files under `roma4d/docs/` when you need a smaller diff or a focused PDF export. |
| **Edition** | Aligned with **Roma4D Elite Reference** toolchain **0.1.0**; monorepo **4DEngine**. |
| **Sister files** | Same content, split for editing: `README.md`, `Install_Guide.md`, `User_Guide.md`, `Dependencies_Guide.md`, `LLM_Guide.md`, `Debugging_Guide.md`, `Errors_Guide.md`, `Shipping_Guide.md`, `Roma4D_Guide.md`. |

---

> *“Give me columns of vec4, rotors that compose, and a compiler that does not lie.”* — That is the contract this master guide unpacks, end to end.

---

## Table of contents

### Front matter

1. [Roma4D Master Guide](#roma4d-master-guide) (this page)
2. [Table of contents](#table-of-contents)

### Parts

| Part | Title | Anchor |
| --- | --- | --- |
| **I** | Documentation map & hub | [Part I](#part-i--documentation-map--hub) |
| **II** | Install guide | [Part II](#part-ii--install-guide) |
| **III** | User guide — vision & capabilities | [Part III](#part-iii--user-guide--vision--capabilities) |
| **IV** | Dependencies & toolchain | [Part IV](#part-iv--dependencies--toolchain) |
| **V** | LLM-assisted development | [Part V](#part-v--llm-assisted-development) |
| **VI** | Debugging | [Part VI](#part-vi--debugging) |
| **VII** | Errors & failure surface | [Part VII](#part-vii--errors--failure-surface) |
| **VIII** | Shipping native products | [Part VIII](#part-viii--shipping-native-products) |
| **IX** | Elite reference specification (normative) | [Part IX](#part-ix--elite-reference-specification-normative) |

### Part IX — major sections (spec)

The elite reference is long; scroll **Part IX** or use in-file search. The spec’s **own numbered table of contents** (§1–§**30**) appears again inside Part IX; **those inline links were written for the standalone `Roma4D_Guide.md`** and may not resolve in this merged file. For guaranteed jump links, open **`Roma4D_Guide.md`** alongside, or search this page for `## 23.` / section titles. **New:** **§30 Spacetime engine foundations (theory manuscript)** deciphers the RomanAILabs PowerShell/Python demos and the Cl(4,0) / Q-tensor narrative in professional language (non-normative for `.r4d` syntax).

**Good search tokens:** `Mental model`, `LLM hard rules`, `Pre-submit checklist`, `Compiler and linker error catalog`, `Native AI Expert`.

---

> **Tip:** In GitHub or VS Code Markdown preview, `Ctrl+F` / `Cmd+F` searches this entire book. For LLM context windows, attach **Part IX** alone (`Roma4D_Guide.md`) when only the spec is needed.

---

## Part I — Documentation map & hub

This folder is the **canonical home** for how-to guides: from **first install** to **shipping native products**, **LLM-assisted development**, and **debugging** when something breaks.

The **[Roma4D Elite Reference Guide](Roma4D_Guide.md)** remains the **normative spec** (what compiles, exact syntax, LLM hard rules). The guides here **route** you to the right section and add **workflow** context.

---

### Start here

| I want to… | Read |
|------------|------|
| **The entire library in one scroll** | You are in it — start at [Table of contents](#table-of-contents), then **Part I** onward. |
| **Install** the toolchain (Windows / Linux / macOS) | **Part II** below, or standalone **[Install_Guide.md](Install_Guide.md)** |
| **Understand** what the language is *for* (vision, applications) | **Part III**, or **[User_Guide.md](User_Guide.md)** |
| **Write code** with correct syntax and rules | **Part IX**, or **[Roma4D_Guide.md](Roma4D_Guide.md)** |
| **See** all tools and versions (Go, Zig, clang, env vars) | **Part IV**, or **[Dependencies_Guide.md](Dependencies_Guide.md)** |
| **Use ChatGPT / Cursor** safely on `.r4d` | **Part V**, or **[LLM_Guide.md](LLM_Guide.md)** |
| **Debug** failed builds and runs | **Part VI**, or **[Debugging_Guide.md](Debugging_Guide.md)** |
| **Decode** compiler / linker messages | **Part VII**, or **[Errors_Guide.md](Errors_Guide.md)** |
| **Ship** executables and releases | **Part VIII**, or **[Shipping_Guide.md](Shipping_Guide.md)** |

---

### Guide map (elaborate track)

#### 1. Install and environment

- **[Install_Guide.md](Install_Guide.md)** — Clone, prerequisites, one-command install, verify `r4d`, troubleshooting checklist, optional 4DOllama.
- **[Dependencies_Guide.md](Dependencies_Guide.md)** — Go, Zig, clang, MSYS2, `R4D_*` variables, `roma4d.toml`, optional Rust for 4DOllama.

#### 2. Use the language

- **[User_Guide.md](User_Guide.md)** — Capabilities, “what can I build,” creative and technical directions (grounded vs roadmap).
- **[Roma4D_Guide.md](Roma4D_Guide.md)** — **Elite reference**: types, `par`, `spacetime:`, ownership, builtins, pipeline, **§20** examples index.

#### 3. LLM-assisted development

- **[LLM_Guide.md](LLM_Guide.md)** — Which doc to paste, protocol, checklists, Expert vs strict.
- **Roma4D_Guide.md** — **§25–§29**: generation protocol, pre-submit checklist, **hard rules**, **Native AI Expert**.

#### 4. Debugging and errors

- **[Debugging_Guide.md](Debugging_Guide.md)** — `R4D_DEBUG`, `last_build_failure.log`, `r4`/`r4d` flags, Expert terminal flow.
- **[Errors_Guide.md](Errors_Guide.md)** — How to read failures; pointer to **Roma4D_Guide §23** (error catalog).

#### 5. Shipping products

- **[Shipping_Guide.md](Shipping_Guide.md)** — `r4 build`, artifacts, naming, bench, CI, licensing, optional container story for services.

---

### Monorepo pointers

| Topic | Location |
|--------|----------|
| Root **INSTALL** stub (quick link) | [../../INSTALL.md](../../INSTALL.md) |
| **4DOllama** product (HTTP API, `4dollama`) | [../../docs/4DOllama.md](../../docs/4DOllama.md) |
| Roma4D **README** (quick start, CLI) | [../README.md](../README.md) |

---

### Contributing to docs

- **Normative behavior** → change **Roma4D_Guide.md** and, if needed, compiler/tests.
- **Onboarding / tone / workflow** → change **User_Guide**, **Install_Guide**, or the hub **README** (this file).
- Keep **cross-links** between guides consistent when adding new pages.


---

## Part II — Install guide

**Audience:** Anyone cloning this monorepo on **Linux, macOS, or Windows**.

This guide installs **Roma4D** (`r4d`, `r4`, `roma4d`) so you can compile and run `.r4d` files.  
Optional: **4DOllama** (Go `4dollama` + Rust `four_d_engine`) is a **separate** step at the end.

**Path convention:** Commands below assume your **current directory is the repository root** — the folder that contains **`roma4d/`**, **`go.mod`** (root), and **`scripts/`**. If you are inside `roma4d/`, `cd ..` first.

**More docs:** See **[Documentation hub (README)](README.md)** for use, LLM workflow, debugging, errors, dependencies, and shipping.

---

### 0. Clone the repository

```bash
git clone https://github.com/RomanAILabs-Auth/4DOllama.git 4DEngine
cd 4DEngine
```

(If you use another remote or folder name, keep using that path below.)

---

### 1. Install prerequisites

#### All platforms

| Need | Why |
|------|-----|
| **[Go 1.22+](https://go.dev/dl/)** | Builds the Roma4D compiler. Must be on `PATH` (`go version` works). |

#### Windows only

| Need | Why |
|------|-----|
| **[Zig](https://ziglang.org/download/)** on `PATH` | Default linker (`zig cc`). Install e.g. `winget install Zig.Zig` and **open a new terminal**. |
| **OR** LLVM **clang** + **MinGW-w64** (e.g. [MSYS2](https://www.msys2.org/)) | Fallback if you do not use Zig. The installer tries to set `R4D_GNU_ROOT` when MSYS is in the usual location. |

You do **not** need Visual Studio for the default Roma4D Windows build (toolchain targets `*-windows-gnu`).

#### Linux

| Need | Why |
|------|-----|
| **`clang`** | Links native binaries. e.g. `sudo apt install clang build-essential` (Debian/Ubuntu). |
| **`git`** | Clone / modules. |

#### macOS

| Need | Why |
|------|-----|
| **Xcode Command Line Tools** or **Homebrew `llvm`** | So `clang` is available. e.g. `brew install go llvm`. |

**Deep dive:** [Dependencies_Guide.md](Dependencies_Guide.md)

---

### 2. One-command install (Roma4D)

Run from the **repository root**.

#### Windows (PowerShell)

**Option A — double-click:** in File Explorer, open your clone folder and double-click **`install.cmd`**. When it says success, **close it** and open a **new** terminal for `r4d`.

**Option B — copy/paste:**

```powershell
cd $HOME\Desktop\4DEngine
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Roma4d.ps1
```

- If your clone is elsewhere, change the first line.
- Skip tests (faster):  
  `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Roma4d.ps1 -SkipTests`

**Important:** When it finishes, **close the terminal and open a new one** so **PATH** and **`R4D_PKG_ROOT`** load.

#### Linux and macOS (bash/sh)

```bash
cd ~/path/to/4DEngine
chmod +x scripts/install-roma4d.sh roma4d/install-full.sh
./scripts/install-roma4d.sh
```

- Skip tests: `SKIP_TESTS=1 ./scripts/install-roma4d.sh`

Add what the script prints to `~/.bashrc` / `~/.zshrc`, **or** run these in **every** new terminal:

```bash
export PATH="$(go env GOPATH)/bin:$PATH"
export R4D_PKG_ROOT="/absolute/path/to/4DEngine/roma4d"
```

Use the **real** path to the folder that contains **`roma4d.toml`**.

---

### 3. Verify Roma4D

Open a **new** terminal (Windows) or ensure `PATH` / `R4D_PKG_ROOT` (Unix).

```bash
r4d version
```

Run the tiny example (exit code **42** is **normal**):

**Windows:**

```powershell
r4d C:\path\to\4DEngine\roma4d\examples\min_main.r4d
```

**Linux / macOS:**

```bash
r4d /path/to/4DEngine/roma4d/examples/min_main.r4d
```

---

### 4. If something fails (checklist)

1. **`go: command not found`** — Install Go; open a **new** terminal.
2. **Windows: scripts disabled** — Use `-ExecutionPolicy Bypass` (§2) or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. **`r4d` not found** — New terminal (Windows); add `GOPATH/bin` to `PATH` (Unix).
4. **Link errors (Windows)** — Install **Zig**, or **clang + MinGW**, then re-run the installer.
5. **Build log** — `roma4d/debug/last_build_failure.log`; set **`R4D_DEBUG=1`**. See [Debugging_Guide.md](Debugging_Guide.md).
6. **Wrong binary** — `where r4d` (Windows) / `which r4d` (Unix).

**More:** [Errors_Guide.md](Errors_Guide.md)

---

### 5. Manual install (without universal scripts)

From **repo root**:

```bash
cd roma4d
```

**Windows:** `.\Install-Full.ps1`  
**Linux / macOS:** `chmod +x install-full.sh && ./install-full.sh`

---

### 6. Optional — 4DOllama (same monorepo, different stack)

**4DOllama:** Ollama-shaped HTTP API + **`4dollama`** Go CLI (default port **13373**). Needs **Rust** + **Go CGO** on Windows for full native engine (stub builds possible without CGO).

- **Windows (repo root):**  
  `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1`
- **Docs:** [../../docs/4DOllama.md](../../docs/4DOllama.md) (from this file: monorepo `docs/` folder).
- **Linux Docker:** `./scripts/install-linux.sh`

Do **not** confuse **`4dollama`** (Go) with **`fourdollama`** (optional Python bridge, port **13377**).

---

### 7. Where things live (repo layout)

| Item | Location (from repo root) |
|------|---------------------------|
| This install guide | `roma4d/docs/Install_Guide.md` |
| Doc hub | `roma4d/docs/README.md` |
| Universal installers | `scripts/Install-Roma4d.ps1`, `scripts/install-roma4d.sh` |
| Roma4D compiler source | `roma4d/` |
| Windows double-click | `install.cmd` |
| Full Roma4D scripts | `roma4d/Install-Full.ps1`, `roma4d/install-full.sh` |
| User PATH + `R4D_PKG_ROOT` (Windows) | `roma4d/scripts/Install-R4dUserEnvironment.ps1` |
| 4DOllama Windows installer | `scripts/install.ps1` |

---

You only need **§0 → §3** for a standard Roma4D setup. The rest is optional or for troubleshooting.


---

## Part III — User guide — vision & capabilities

**Roma4D** is a **compiled**, **native** programming language built around **four-dimensional geometric algebra** (**Cl(4,0)**), **structure-of-arrays (SoA) data**, and **explicit parallelism**—with a surface that reads like **Python** so ideas move fast from brain to binary.

This guide is the **“why” and “what if”** story. For exact syntax, rules, and what compiles today, use the **[Roma4D Elite Reference Guide](Roma4D_Guide.md)**.

---

### Why bother with 4D in a programming language?

Most code treats the world as lists of numbers and ad-hoc matrices. That works until you need **rotations that compose cleanly**, **planes and volumes** in the same algebra as **vectors**, or **massive parallel updates** over millions of entities without pointer soup.

**4D here is not a gimmick.** It is a **mathematical workspace** where:

- **Vectors** (`vec4`) carry position, direction, color, or any four-tuple you want to treat geometrically.
- **Rotors** encode rotations in planes; they **multiply** like the physics of the real world expects.
- **Multivectors** hold the full Clifford algebra—grades mixed intentionally—so you can express **fields**, **blades**, and **operators** in one type system instead of bolting quaternions, matrices, and tensors together by hand.

Roma4D makes that algebra **first-class**: operators like **`*`**, **`^`**, and **`|`** mean geometric things, not “whatever overload C++ felt like today.”

---

### What Roma4D is *amazing* at (vision + reality)

Below is a deliberately **wide** lens—applications that fit the **mental model** (columns, rotors, parallel sweeps, spacetime-shaped programs). Some demos exist in this repo today; others are **natural targets** as the toolchain and GPU path mature. When something is still **roadmap**, we say so.

#### 1. Living universes — millions of entities, one coherent geometry

Think **N-body–style** worlds, **particle galaxies**, **crowds**, **sensor swarms**, or **game entities** whose state is fundamentally “a bag of vectors that evolve under the same rules.”

Roma4D wants you to store hot state as **`list[vec4]`** and **SoA fields**, then sweep them with **`par for`**. That is not just “faster loops”—it is an invitation to write **physics as geometry**: each step is **rotate**, **scale**, **add**, **project**, repeated at scale, with the compiler aiming at **SIMD** and (over time) **GPU-oriented** lowering for `spacetime:` regions.

**Crazy-but-sane idea:** a **cosmic-scale** demo where ten million worldlines are not a stunt—they are a **stress test** that the language layout (SoA + par) was designed to survive.

#### 2. “Spacetime-shaped” programs (compile-time narrative)

Roma4D has **`t`**, **`expr @ t`**, and **`spacetime:`** blocks. Today they primarily drive **compile-time staging** and **MIR metadata**—they shape *how* the compiler thinks about regions, not a hidden interpreter loop in your release binary.

**Why that still matters creatively:** you can **author** programs where *when* a quantity is meaningful is **written in the source**, not scattered across booleans and phase flags. As the pipeline grows, that becomes the backbone for **temporal optimization**, **GPU scheduling hints**, and **proof-like structure** in simulation code.

**Outside-the-box:** treat **`spacetime:`** as a **director’s mark** in code—“this whole region is one physical scene transition”—even before every backend exploit is switched on.

#### 3. Robotics, aerospace, and anything that rotates for a living

Where traditional code chains **Euler angles** and prays, geometric algebra **composes rotors** without gimbal melodrama. Roma4D is a strong fit for:

- **Joint chains** and **constraints** expressed as **planes and rotors**.
- **Sensor fusion** where **frames** are **transforms**, not accidental 4×4 soup.
- **Satellite / drone** attitude and **reference frames** as **first-class rotors**.

If your bug sounds like “the rotation order is wrong on Tuesdays,” you are the target user.

#### 4. Creative coding and “impossible” visuals

**Projective `w` in `vec4`** is a classic graphics trick; here it sits inside an algebra that also knows **outer products** and **contractions**. That opens:

- **Non-Euclidean** toy worlds (warped stepping, weird lights, “4D→2D” projections as experiments).
- **Shader-like logic** experimented with in **native** code first, then ported once you like the math.
- **Generative art** where **fields** are multivector-valued and **par** smears them across a grid.

You are not stuck in “triangle API of the month”—you are sketching **geometry** and letting LLVM carry it.

#### 5. Signal and field metaphors (Clifford-flavored DSP)

Multivectors can represent **structured measurements**—not just audio samples, but **vector sensor arrays**, **polarized** quantities, **mixed-grade** decompositions. **`par for`** then becomes “update every probe the same way, deterministically.”

**Wild-but-plausible:** compressive sensing, beamforming, or custom **geometric filters** where a **blade** is not a metaphor but a **typed value**.

#### 6. Science at the boundary of geometry and data

The repo includes **large, serious demos** (e.g. **cosmic genesis**, **collider**, **proteome**-themed experiments) that are valid Roma4D—not Python scripts pretending. That spirit is the point: **express huge structured models** in a language that **won’t silently dynamic-type you out of performance**.

**Honest frontier:** bleeding-edge HPC and CUDA-backed execution are **roadmap**; today you get **native CPU** executables, **LLVM**, and **hooks** that record intent for what comes next. The **language** is not waiting—the **runtime story** is catching up.

#### 7. Systems programmers who hate mystery allocations

**Ownership 2.0** and **SoA linear access** mean: hot paths are **predictable**. You **move** values out of columns, **transform** them with geometric ops, **write back**. Parallel regions have **explicit** capture discipline.

**Amazing consequence:** you can pitch Roma4D to people who love **Rust’s honesty** but need **rotors** in the type system—not in a side library.

#### 8. Teaching “real math” that actually runs

Because the surface looks like Python, you can **teach** **rotors**, **blades**, and **parallel sweeps** without students drowning in template errors first. Then **`r4d`** hands them a **binary**—confidence that the idea is not only notation.

---

### The emotional pitch (yes, really)

**3D graphics** taught a generation to think in matrices. **ML** taught a generation to think in tensors with opaque autodiff. **Roma4D** invites you to think in **geometry that composes**—and to **compile** that thought into something **fast** and **inspectable**.

If you have ever felt that “my simulation is correct but the math is spread across seventeen files,” this language is trying to **pull the truth back into one algebra**—and then **let the machine parallelize** the honest structure you wrote.

---

### What to run first

| You want… | Start here |
|-----------|------------|
| Smallest working program | `examples/min_main.r4d` |
| Hello geometry | `examples/hello_4d.r4d` |
| Spacetime + `par` shape | `demos/spacetime_collider.r4d` |
| Epic scale showcase | `demos/cosmic_genesis.r4d` |
| Install toolchain | **[Install_Guide.md](Install_Guide.md)** (full) · root [INSTALL.md](../../INSTALL.md) (quick link) |

---

### Relationship to other docs

| Document | Role |
|----------|------|
| **[Roma4D_Master_Guide.md](Roma4D_Master_Guide.md)** | **All guides in one file** (you are reading it) |
| **This User Guide** (Part III) | Capabilities, vision, “what can I build?” |
| **[Roma4D_Guide.md](Roma4D_Guide.md)** | Same text as **Part IX** — use standalone file for working anchor links |
| **[README.md](README.md)** (this docs folder) | Hub: install, LLM, errors, shipping, dependencies |
| **[../README.md](../README.md)** | Repo quick start, CLI table, layout |

---

### One-line summary

**Roma4D lets you write native programs where four-dimensional geometric algebra, parallel columns, and spacetime-shaped structure are the main character—not an afterthought library.**

Welcome to the workshop. Build something unreasonable.


---

## Part IV — Dependencies & toolchain

What must be on your machine, what is optional, and which **environment variables** the toolchain respects.

---

### Roma4D compiler (`r4d`, `r4`, `roma4d`)

#### Required

| Dependency | Version / notes |
|------------|-----------------|
| **Go** | **1.22+** — builds the driver from `roma4d/cmd/`. `go version` must work. |

#### Windows — native linking (pick at least one path)

| Dependency | Role |
|------------|------|
| **Zig** | **Default.** `zig` / `zig.exe` on `PATH`; driver invokes `zig cc`. Override binary with **`R4D_ZIG`** (full path to executable). |
| **LLVM clang + MinGW** | Fallback when Zig is absent. Target **`*-windows-gnu`**. **`R4D_GNU_ROOT`** — MinGW prefix (installer may set from MSYS2 paths). |

You typically do **not** need the MSVC toolchain for Roma4D’s default Windows pipeline.

#### Linux

| Dependency | Role |
|------------|------|
| **clang** | Link step; math via `-lm` (handled by driver when linking `rt/roma4d_rt.c`). |
| **build-essential** (or distro equivalent) | C toolchain for the small runtime. |

#### macOS

| Dependency | Role |
|------------|------|
| **clang** | Via Xcode CLT or **Homebrew `llvm`**. |
| **Go** | Same as above. |

---

### Environment variables (Roma4D)

| Variable | Purpose |
|----------|---------|
| **`R4D_PKG_ROOT`** | Directory containing **`roma4d.toml`** — set by **Install-R4dUserEnvironment.ps1** (Windows user env). Lets **`r4d /any/path/file.r4d`** resolve imports without `cd` into the clone. |
| **`R4D_ZIG`** | Full path to **Zig** if `zig` is not on `PATH`. |
| **`R4D_GNU_ROOT`** | MinGW/MSYS prefix for **clang** fallback on Windows (headers, `mm_malloc.h`, etc.). |
| **`R4D_DEBUG`** | Non-empty → extra diagnostics (mirrors / supplements `last_build_failure.log` per driver behavior). |

Embedded root: binaries from **`go install -ldflags "-X …EmbeddedPkgRoot=…"`** can bake the repo path so **`R4D_PKG_ROOT`** is optional when using those builds.

---

### Project manifest

| File | Role |
|------|------|
| **`roma4d.toml`** | Package root marker; **`[build] backend = "llvm"`**, edition, etc. The compiler walks **upward** from a `.r4d` file until it finds this file. |

---

### 4DOllama stack (optional, monorepo root)

Not required to **compile Roma4D**. Needed only if you run the **HTTP / GGUF** product.

| Dependency | Role |
|------------|------|
| **Rust + cargo** | Builds **`4d-engine`** (`four_d_engine` staticlib/cdylib). |
| **Go + CGO** | Links **`4dollama`** against `libfour_d_engine` (native path). **`CGO_ENABLED=0`** → stub engine (no full native 4D FFI). |
| **gcc / MSVC** | C compiler for CGO on your platform. |

See **[../../docs/4DOllama.md](../../docs/4DOllama.md)** and root **`scripts/install.ps1`**.

---

### Python `fourdollama` (optional bridge)

Separate from **`4dollama`** (Go). Uses **FastAPI** on **13377** by default. Needs **Python 3.10+** and **`pip install -e` from `4DOllama/`**. Optional **`r4d`** on `PATH` for local kernel runs.

---

### Quick verification

```bash
go version
## Windows:
zig version
## or: clang --version
## Linux/macOS:
clang --version
r4d version
```

**Install:** [Install_Guide.md](Install_Guide.md) · **Ship binaries:** [Shipping_Guide.md](Shipping_Guide.md)


---

## Part V — LLM-assisted development

Use this page to **paste the right document** into ChatGPT, Cursor, Copilot, etc., and to follow a **safe generation protocol** so the model emits **valid `.r4d`**, not “Python with 4D comments.”

---

### Golden rule

**Roma4D is not Python.** It is **ahead-of-time compiled** to **native code** via **LLVM**. There is **no** CPython, **no** f-strings, **no** `time()` wall clock, **no** arbitrary PyPI imports.

If an LLM drifts into Python, **stop** and re-paste **[Roma4D_Guide.md](Roma4D_Guide.md)** §**Mental model** and §**27 LLM hard rules**.

---

### What to attach / paste

| Goal | Primary document |
|------|------------------|
| **Generate or refactor `.r4d`** | **[Roma4D_Guide.md](Roma4D_Guide.md)** (full file if the tool allows — it is the spec). |
| **High-level “what is this language?”** | **[User_Guide.md](User_Guide.md)** + **Roma4D_Guide § Mental model**. |
| **Install / PATH / `r4d` not found** | **[Install_Guide.md](Install_Guide.md)**. |

Minimum for codegen: **Roma4D_Guide** sections **Mental model**, **Types**, **Builtins**, **`par`**, **`spacetime:`**, **Ownership 2.0**, **§22 Python vs Roma4D**, **§26 Pre-submit checklist**, **§27 Hard rules**.

---

### Step-by-step protocol (summary)

Full detail: **Roma4D_Guide §25** — *LLM code-generation protocol*.

1. Decide **entry shape**: `main() -> None` vs `-> int`.
2. Prefer **`list[vec4]` + `par for`** for parallel 4D simulation shapes unless the spec needs something smaller.
3. Use **only** APIs that appear in the Guide or in **`examples/`** / **`demos/`** indexed in **§20**.
4. **Strings:** static literals for `print` in portable programs; do not format numbers into strings in source.
5. **Time:** **`t`** / **`@ t`** are **compile-time staging**, not a runtime clock.
6. After drafting, run the **§26 pre-submit checklist** line by line.
7. Run **`r4d yourfile.r4d`** or **`r4 run --strict`** for CI-style raw errors.

---

### Native AI Expert (terminal)

When **`r4d`** / **`r4 run`** fails in **forgiving** mode, the toolchain can show a **structured debug block** (copy-paste friendly), optional hints, and a small **interactive** Q&A on a TTY.

**Spec:** **Roma4D_Guide §29** — *Native AI Expert*.

**CI / automation:** use **`r4 run --strict`** (or equivalent) so output stays **machine-parsable** without interactive prompts.

---

### Prompt snippets you can reuse

**System / preface:**

> You are generating **Roma4D** source (extension `.r4d`). It is **not** Python. Follow the attached **Roma4D Elite Reference Guide** exactly. If an API is not in the guide or indexed examples, do not invent it.

**After a bad generation:**

> Re-read **§27 LLM hard rules** and **§22 invalid patterns**. Rewrite the smallest program that compiles; use **`examples/min_main.r4d`** as a template.

---

### Related

- **Debugging:** [Debugging_Guide.md](Debugging_Guide.md)  
- **Errors:** [Errors_Guide.md](Errors_Guide.md)  
- **Hub:** [README.md](README.md)


---

## Part VI — Debugging

When **`r4d`**, **`r4 run`**, or **`r4 build`** fails, use this flow before opening an issue or asking an LLM.

---

### 1. Reproduce with clear flags

| Command | When to use |
|---------|-------------|
| **`r4d file.r4d`** | Default: forgiving mode may invoke **Native AI Expert** hints on failure. |
| **`r4 run --strict file.r4d`** | **CI / logs** — raw compiler errors, no interactive fluff. |
| **`r4 run -bench file.r4d`** | See **per-phase timings** (lexer → link → **native_run**). |
| **`R4D_DEBUG=1`** (Unix) or set user env on Windows | Extra stderr diagnostics where implemented. |

---

### 2. Read the build failure log

On many failures the driver writes:

**`roma4d/debug/last_build_failure.log`**

It typically captures:

- Linker command line (**`zig cc`** or **`clang`**)  
- **stderr** from the linker  
- A **head** of generated **LLVM IR** (useful for codegen bugs)

Open this file **first** when the terminal output scrolls away.

---

### 3. Common classes of failure

| Symptom | Check |
|---------|--------|
| **Linker not found** | **Zig** or **clang** on `PATH`; **`R4D_ZIG`**; **`R4D_GNU_ROOT`** (Windows clang path). |
| **`roma4d.toml` not found** | Run from a tree that contains **`roma4d.toml`** upward from the `.r4d` path, or set **`R4D_PKG_ROOT`**. |
| **Type / ownership errors** | **Roma4D_Guide §11** Ownership 2.0; **§10** SoA / aos. |
| **Illegal Python-ism** | **Roma4D_Guide §22** — f-strings, dynamic imports, `while` growth patterns, etc. |

---

### 4. Native AI Expert (interactive)

In default **forgiving** mode, the CLI may print a **rich debug block** and offer **patch hints** or Q&A.

**Full behavior:** **Roma4D_Guide §29**.

For **scripts and CI**, always prefer **`--strict`**.

---

### 5. LLVM / MIR introspection

Advanced:

- Inspect **MIR** and **LLVM IR** expectations in **`roma4d/tests/`** and compiler sources.  
- **`-bench`** helps separate **compile** vs **run** time.

---

### 6. Related docs

- **Error message catalog:** **Roma4D_Guide §23** — also summarized in [Errors_Guide.md](Errors_Guide.md).  
- **Install / PATH:** [Install_Guide.md](Install_Guide.md).  
- **LLM-assisted fix:** [LLM_Guide.md](LLM_Guide.md).


---

## Part VII — Errors & failure surface

This guide **orients** you; the **authoritative catalog** lives in the Elite Reference.

---

### Authoritative reference

**[Roma4D_Guide.md — §23 Compiler and linker error catalog](Roma4D_Guide.md)** (search for `## 23.` in-file)

Use that section when mapping a **specific** message to a **fix**.

---

### Failure surface (what you might see)

| Layer | Typical source |
|-------|----------------|
| **Lexer / parser** | Invalid syntax, wrong extension, BOM issues — see Guide §6–§8. |
| **Typecheck** | Wrong types, missing builtins, bad `par` capture — §7, §11, §13. |
| **Ownership 2.0** | SoA move/borrow violations — §11. |
| **MIR / LLVM** | Internal assert or bad lowering — often needs `last_build_failure.log` + issue upstream. |
| **Linker** | Missing **Zig** / **clang**, wrong **Windows** ABI, missing **`-lm`** (usually handled by driver on Unix). |

---

### Install-time vs compile-time

| Problem | Doc |
|---------|-----|
| **`go` / `r4d` not found** | [Install_Guide.md §4](Install_Guide.md#4-if-something-fails-checklist) |
| **`zig` / `clang` missing** | [Dependencies_Guide.md](Dependencies_Guide.md) |
| **Script execution policy (Windows)** | [Install_Guide.md](Install_Guide.md) |

---

### Workflow

1. Capture **full** output with **`r4 run --strict`** (or save terminal log).  
2. Open **`roma4d/debug/last_build_failure.log`** if present.  
3. Match tokens to **Roma4D_Guide §23**.  
4. If stuck, paste **§23** + your log into an LLM **with** [LLM_Guide.md](LLM_Guide.md) constraints.

**Debugging workflow:** [Debugging_Guide.md](Debugging_Guide.md)


---

## Part VIII — Shipping native products

Roma4D emits **native executables** (e.g. **`.exe`** on Windows). There is **no** separate runtime VM—what you ship is the **binary** plus any **assets** you bundle yourself.

---

### 1. Produce a release binary

From the **package root** (directory with **`roma4d.toml`**), typical flow:

```bash
r4 build path/to/app.r4d -o path/to/myapp.exe
```

(On Unix, use your desired output name without `.exe` if you prefer.)

| Flag / behavior | Purpose |
|-----------------|--------|
| **`-o path`** | Persistent artifact; default is often next to source with platform suffix. |
| **`-bench`** | Compile-phase timings (see README / Roma4D_Guide CLI table). |
| **`--strict`** | Fail without Expert UI — good for **CI** builds. |

**CLI reference:** [../README.md](../README.md) (CLI table) or **Roma4D_Guide §1**.

---

### 2. What to ship

| Item | Notes |
|------|--------|
| **Executable** | Single file for many programs; statically linked runtime pieces depend on linker (Zig/clang) and `roma4d_rt.c`. |
| **Data files** | Ship alongside the binary; load via paths you control (no special Roma4D package format required for assets). |
| **License** | Include **LICENSE** from the repo if you redistribute compiler output; respect **third-party** terms of linked runtimes (LLVM, libc, etc.). |

---

### 3. Continuous integration

- **Install:** [Install_Guide.md](Install_Guide.md) — on Linux runners use **`clang`** + **Go**.  
- **Build:** `r4 build --strict your_ci_entry.r4d -o artifact`  
- **Test:** `go test ./...` under **`roma4d/`** validates the **compiler**; your **app** should have its own **`r4 run --strict`** smoke test.

Monorepo **CI** example: see root **`.github/workflows/ci.yml`** (Roma4D + Go modules).

---

### 4. Cross-compilation

Today the **host** toolchain (Zig/clang) defines targets. For exotic targets, align **triple** and linker with your **Zig** / **clang** setup and project policies—consult **Roma4D_Guide §1** (linker) and compiler sources before relying on a specific cross pair in production.

---

### 5. Shipping *services* (4DOllama)

If your “product” is an **API** rather than a CLI tool:

- Build / run **`4dollama serve`** per **[../../docs/4DOllama.md](../../docs/4DOllama.md)**.  
- **Docker:** root **`scripts/install-linux.sh`** + `docker run` pattern.  
- **Ports:** default **13373** (Go) vs **13377** (optional Python bridge)—document for users.

---

### 6. Versioning and reproducibility

- Pin **Go** and **Zig**/**clang** versions in CI (matrix or container image).  
- Commit **`go.sum`** / lockfiles for **reproducible** compiler builds from source.  
- Record **Roma4D** toolchain **version** (`r4d version`) in release notes.

---

### Related

- **Dependencies:** [Dependencies_Guide.md](Dependencies_Guide.md)  
- **Debug failed release builds:** [Debugging_Guide.md](Debugging_Guide.md)  
- **Doc hub:** [README.md](README.md)


---

## Part IX — Elite reference specification (normative)

> **This part is the authoritative language spec** — toolchain 0.1.0. If a split guide and this master disagree, treat **Part IX** and the standalone `Roma4D_Guide.md` as source of truth.

**Audience:** Expert practitioners and **automated code generators** (LLMs). Treat this document as a **normative specification** for what **compiles and links today** in toolchain **0.1.0**. If something is not described here as supported, **do not emit it**—prefer a smaller, boring program that builds.

**How-to hub (install, use, LLM workflow, debugging, shipping):** [README.md](README.md)

**Official source extension:** **`.r4d`**. Legacy **`.r4s`** and **`.roma4d`** are still accepted by the driver. **`.py` is not Roma4D.**

**Edition:** `2025` in `roma4d.toml` names the staged feature set.

**Not Python:** Roma4D uses **Python 3.12–shaped surface syntax** (indentation, `def`, `class`, many keywords) but is a **separate, ahead-of-time compiled** language targeting **LLVM IR → native executable**. There is **no** CPython interpreter, **no** arbitrary `import` of PyPI packages, **no** f-strings, **no** `time()` wall clock, **no** dynamic `eval`.

---

#### How an LLM should use this document

1. **Read [Mental model](#mental-model-read-this-first) and [LLM hard rules](#27-llm-hard-rules-non-negotiable) first.** If the user asks for “Python with 4D,” **refuse** and output valid **`.r4d`** per this spec.
2. **Before writing code**, decide: entry type `main() -> None` vs `-> int`, whether imports are needed (`from libgeo import ...`), and whether the program uses **`list[vec4]` + `par for`** (safe default for “4D simulation” asks).
3. **After drafting**, run the **[pre-submit checklist](#26-pre-submit-checklist-copy-for-every-generation)** mentally line-by-line.
4. **On uncertainty**, choose the **smallest** construct that appears in **`examples/hello_4d.r4d`** or **`demos/spacetime_collider.r4d`**. Never invent APIs.
5. **Strings:** only **static** string literals for `print` in portable programs. Do not format numbers into strings in Roma4D source.
6. **Time:** **`t`** and **`expr @ t`** are **compile-time staging** tokens—not a runtime clock.

---

#### Table of contents

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

---

#### Mental model (read this first)

Roma4D is built around three core ideas:

1. **Columns, not heap objects**  
   Hot data is expressed as **SoA fields** and **`list[vec4]` worldtubes** scanned by **`par for`**, not pointer graphs.

2. **Time is explicit in the language**  
   **`t`**, **`expr @ t`**, and **`spacetime:`** blocks express *when* (compile-time narrative / MIR staging). They **do not** add a hidden interpreter loop in the emitted binary for ordinary programs.

3. **Parallelism is first-class**  
   **`par for`** marks structured parallel loops with ownership-aware capture rules and backend hints (SIMD / future GPU paths).

If you model problems as **vectors + rotors + parallel updates**, you stay on the supported path.

---

#### 1. How the compiler runs

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

#### 2. Project layout and `roma4d.toml`

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

#### 3. CLI commands, PATH, and package root

##### Simplest usage (remember this)

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

##### Package root resolution (`findPkgRoot`)

Order of resolution:

1. Walk **upward** from the **source file’s directory** until **`roma4d.toml`** is found.
2. If not found, try **`R4D_PKG_ROOT`**, then **`ROMA4D_HOME`** (must contain `roma4d.toml`).
3. If still not found, use **`EmbeddedPkgRoot`** (set at **link time** when built via `install.ps1`, `install.sh`, or `scripts/Install-R4dUserEnvironment.ps1` with `-ldflags -X .../cli.EmbeddedPkgRoot=...`).

This is how **`r4d C:\Desktop\foo.r4d`** works **without** `cd` into the clone after a proper install.

##### Relative paths and cwd

- **`r4 run demos\foo.r4d`** resolves **`demos\foo.r4d`** relative to the **shell current working directory**, not relative to `R4D_PKG_ROOT`.
- **Imports** (`from libgeo import ...`) resolve under **package root**, not next to the sketch file.

##### Windows PowerShell

Do **not** paste the **`PS C:\...>`** prompt into the terminal; on some systems **`PS`** is an alias and breaks the line.

##### Expert mode (forgiving, default)

On **`.r4d` build/run failure**, the CLI invokes the **Native AI Expert** (`src/ai/expert.go`): a **copy-pasteable terminal debug block**, **`debug/last_build_failure.log`** append, optional **patch suggestion + [y/N] prompt**, and an **interactive** Q&A when **stdin is a TTY**. Full behavior is specified in **§29**.

---

#### 4. Modules and imports

##### 4.1 Module file resolution

For `import foo.bar` or `from foo.bar import x`, the driver tries paths under **package root** (see `ResolveRoma4DModuleFile` in `src/compiler/source_ext.go`):

**Per segment / dotted name, extension order is always:** **`.r4d` → `.r4s` → `.roma4d`**.

Examples (conceptual):

- `pkgRoot/foo/bar.r4d` (or `.r4s` / `.roma4d`)
- `pkgRoot/foo.bar.r4d` (dotted flat file)

Missing file → **`ImportError`**.

##### 4.2 Import forms

| Form | Supported |
|------|-----------|
| `import mod` | Yes (with optional `as`). |
| `from mod import a, b, c` | Yes. |
| `from mod import *` | **No** — hard error in typechecker. |

##### 4.3 Exports (v0 behavior)

Importable names are **top-level** **`def`** and **`class`** in the module file. Prefer defining public APIs at module scope.

##### 4.4 Example library (`libgeo.r4d`)

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

#### 5. Entry point, functions, and control flow

##### 5.1 `main`

- Entry point is **`def main()`** (no `if __name__ == "__main__"`).
- **`def main() -> int:`** must **`return`** an integer on all paths (otherwise **synthesized return** warning).
- **`def main() -> None:`** may use **`return None`**, bare **`return`**, or fall off the end; lowering targets **void** `main` in LLVM.

##### 5.2 Function definitions

- Use **`def name(args) -> ReturnType:`** with **annotated** parameters and return for public / non-trivial functions.
- **Variadic** Python features beyond what examples use: **avoid** unless you have verified with `r4d --strict`.

##### 5.3 Statements (supported idioms)

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

#### 6. Lexical structure: tokens, keywords, literals

##### 6.1 Comments and docstrings

- **`#` line comments** anywhere.
- **Triple-quoted strings** used as **class docstrings** appear in sources (e.g. `hello_4d.r4d`); treat as documentation only.

##### 6.2 Identifiers

Standard Python-like identifiers.

##### 6.3 Numeric literals

- **Integers** and **floats** as in Python.
- **Underscore separators** allowed: **`1_000_000`** (see demos).

##### 6.4 String literals

- **`"..."` and `'...'`** only.
- **No f-strings** (`f"..."`) — **not** the string model.
- **No** automatic formatting of **`int` / `float`** for human display in portable Roma4D—use **fixed messages** (`print("tick 512")`).

##### 6.5 Keywords (representative set)

Includes Python keywords **plus** Roma4D extensions:

**Roma4D-specific (non-negotiable spelling):** `par`, `soa`, `aos`, `vec4`, `rotor`, `multivector`, `borrow`, `mutborrow`, `unsafe`, **`t`** (time coordinate), **`spacetime`**.

**Also reserved / lexed as keywords:** `def`, `class`, `return`, `if`, `else`, `elif`, `for`, `while`, `pass`, `break`, `continue`, `import`, `from`, `as`, `True`, `False`, `None`, `and`, `or`, `not`, `in`, `is`, `try`, `except`, `raise`, `with`, `async`, `await`, `lambda`, `match`, `case`, …

**LLM rule:** If unsure whether a Python keyword works end-to-end, **do not use it**—stick to constructs in **`examples/hello_4d.r4d`**.

---

#### 7. Types (complete table)

##### 7.1 Host / scalar types

| Type | Notes |
|------|------|
| `int` | Signed integer. |
| `float` | Double-precision float in native lowering. |
| `str` | String; **`print`** is oriented toward **string** arguments in the Roma4D runtime. |
| `bool` | `True` / `False`. |
| `none` | Type of **`None`**; also used where no value is returned. |

##### 7.2 4D / systems types

| Type | Role |
|------|------|
| `vec4` | 4-component vector; geometric ops with `rotor` / `multivector`. |
| `rotor` | Rotor value; construct with **`rotor(angle=..., plane="...")`**. |
| `multivector` | General multivector; **`multivector()`** default ctor in examples. |
| `rawptr` | Raw pointer for `unsafe` + **`mir_*`**. |
| `time` | Type of **`t`** and time-coordinate reasoning. |

##### 7.3 Constructors / generics in surface syntax

- **`list[T]`** — e.g. **`list[vec4]`**, **`list[int]`**.
- **Do not** use C/Rust array syntax like **`[T; N]`** for bulk data.

---

#### 8. Builtins and constructors (authoritative)

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

#### 9. Geometric algebra (Cl(4,0)) operators

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

#### 10. Classes, `soa`, and `aos`

```roma4d
class Particle:
    soa pos: vec4
    soa vel: vec4
```

- **`soa`** declares **column / linear** fields with Ownership 2.0 semantics (§11).
- **`aos`** exists as a layout keyword; **prefer `soa`** unless you have a verified pattern.

**Construction:** **`Particle()`** lowers to a runtime symbol (see `roma4d_rt.c`).

---

#### 11. Ownership 2.0 (linear SoA, borrows, taint)

**All enforcement is compile-time.** The native binary does not execute a Rust-style borrow checker at runtime.

##### 11.1 Linear `soa` field access (canonical pattern)

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

##### 11.2 `borrow`, `mutborrow`, `timetravel_borrow`

- Operand must be a **simple identifier** matching the borrow pass expectations.
- **`timetravel_borrow(rotor)`** appears inside **`spacetime:`** in **`hello_4d.r4d`** as a chronology narrative hook.

---

#### 12. Spacetime: `t`, `@ t`, `spacetime:`

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

#### 13. Parallelism: `par for`

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

#### 14. Lists and comprehensions

**Construction**

```roma4d
xs: list[vec4] = [vec4(x=0, y=0, z=0, w=1) for _ in range(N)]
```

**Indexing**

- **`xs[i]`** reads an element.
- Updates inside **`par for p in xs`** should use the **`p = ...`** pattern, not index loops, unless you **verify**.

**LLM rule:** **`list[vec4]`** scales with memory—**1_000_000** is the canonical demo scale; **10_000_000** appears in **`demos/cosmic_genesis.r4d`** and is heavier.

---

#### 15. Systems: `unsafe:` and MIR hooks

```roma4d
unsafe:
    rawp: rawptr = mir_alloc(128)
    mir_ptr_store(rawp, 42)
    _peek: int = mir_ptr_load(rawp)
```

Requires **`[systems] unsafe = true`**. **`gc = false`**—no tracing GC.

---

#### 16. Native runtime (`rt/roma4d_rt.c`)

Linked from **`pkgRoot/rt/roma4d_rt.c`**. Provides C symbols used by LLVM lowering, including **`print` → `puts`**, **`vec4`**, **`rotor`**, **`multivector`**, **`Particle`**, and demo hooks calling **`curl`** for Ollama.

**Numeric printing:** not a general **`printf` bridge** from Roma4D for arbitrary **`float`** formatting—use **fixed strings** or extend C if you need digits.

---

#### 17. Ollama / HTTP demos (builtins)

##### `ollama_demo()`

- Requires **`curl`** on `PATH` and local **Ollama** (`ollama serve`, default `http://127.0.0.1:11434`).
- JSON body is **fixed in C** (dynamic string building from `.r4d` is not the general story).

##### `quantum_server_demo()`

- Same infrastructure.
- Env: **`QUANTUM_QUERY`**, **`QUANTUM_CONTINUE=1`** (see `demos/quantum_server.r4d`).

**Security:** uses **`system("curl ...")`** from C—trusted local dev only.

---

#### 18. Compilation pipeline

1. Parse primary + imported modules.
2. Typecheck + resolve imports + attach builtins.
3. Ownership 2.0 pass.
4. Lower AST → MIR.
5. MIR → LLVM IR.
6. **`zig cc` or `clang`** compile `.ll` → `.o`, link **`roma4d_rt.c`** → executable.

---

#### 19. Debugging, logs, and common failures

##### `debug/last_build_failure.log`

On failure, a structured record is appended under **package root** **`debug/last_build_failure.log`**: stage, argv, stderr, LLVM IR head. In **forgiving** mode, the **Native AI Expert** also appends a block under stage **`ai_expert`** (see **§29**).

##### `R4D_DEBUG=1`

Mirrors failure details to stderr immediately (Unix / Windows / PowerShell).

##### Zig vs Clang (Windows)

- **Install Zig** and put it on **`PATH`**—simplest path.
- **Clang fallback:** if you see **`mm_malloc.h`** errors, set **`R4D_GNU_ROOT`** to your MinGW prefix (e.g. `C:\msys64\ucrt64`) or **install Zig**.

##### Linux `sqrt` undefined

Preserve **`-lm`** when customizing links; the driver adds it on non-Windows.

##### Indentation errors

Mixed tabs/spaces cause **`INDENT` / `DEDENT`** parse failures—reformat uniformly.

---

#### 20. Example programs (indexed)

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

#### 21. File extensions

| Ext | Status |
|-----|--------|
| **`.r4d`** | **Official** |
| **`.r4s`** | Legacy |
| **`.roma4d`** | Legacy |

---

#### 22. Python vs Roma4D — invalid pattern encyclopedia

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

#### 23. Compiler and linker error catalog

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

#### 24. Ergonomics: `r4`, `GOBIN`, `R4D_PKG_ROOT`, embedded root

- **`go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d`** puts binaries in **`go env GOBIN`** if set (e.g. conda **`…\Scripts`**), else **`GOPATH\bin`**.
- **`scripts/Install-R4dUserEnvironment.ps1`** (recommended on Windows): appends that **install directory** to **user** **`PATH`**, sets **`R4D_PKG_ROOT`**, optionally **`R4D_GNU_ROOT`** for Clang+MinGW, and runs **`go install`** with **`-ldflags -X …cli.EmbeddedPkgRoot=…`** so **`r4d C:\anywhere\sketch.r4d`** works after a new terminal.
- **`install.ps1`** / **`install.sh`**: same **embedded root** pattern for developers who **`go install`** from the repo.
- **`r4d.ps1`** / **`r4d.cmd`**: developer launcher at repo root—**`Set-Location $PSScriptRoot`**, **`go build`** into Go bin, prepend **`PATH`**, then **`& r4d.exe @args`**. No args → prints a short usage box and **`r4d help`**.
- Plain **`go install`** without **`-ldflags`**: **no embedded root**—either keep sources under a tree that contains **`roma4d.toml`** or set **`R4D_PKG_ROOT`** / **`ROMA4D_HOME`**.
- **CLI errors:** “file not found” suggests a **full path** example; “could not find Roma4D installation” points at **Install-R4dUserEnvironment.ps1** and **`r4d yourfile.r4d`**.

---

#### 25. LLM code-generation protocol (step-by-step)

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

#### 26. Pre-submit checklist (copy for every generation)

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

#### 27. LLM hard rules (non-negotiable)

1. Output **only** **`.r4d`** (unless user explicitly requests legacy extensions).
2. **Never** claim CPython semantics for **`print`**, **`str.format`**, **`time()`**, or **f-strings**.
3. **Never** invent **stdlib** modules beyond local **`.r4d`** files in the package root.
4. **Always** use **`vec4` / `rotor` / `multivector`** for geometric code—never **`Vec4` / `float4`**.
5. **`main() -> int`** must end with an **integer `return`** on all paths.
6. If unsure about a feature, **omit** it—**`print("ok")`** beats a hallucination.
7. After code, instruct: **`r4d <file>.r4d`**, **`--strict`** for CI, **`R4D_DEBUG=1`**, and **`last_build_failure.log`**.

---

#### Appendix A — Drop-in safe template (verified shape)

Copy this when the user gives no extra constraints. It matches **`examples/hello_4d.r4d`** minus optional extras:

```roma4d
### sketch.r4d — minimal safe Roma4D scaffold (adjust N for memory)
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

#### 28. Document maintenance

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

#### 29. Native AI Expert (terminal debug + interactive)

**What it is:** A **built-in**, **zero-network** assistant that runs when **`r4d` / `roma4d` / `r4`** fails to compile or link a **`.r4d`** file in **forgiving mode** (the default). It does **not** call external LLMs; it uses curated rules aligned with this guide.

##### 29.1 Terminal debug block (always in forgiving mode)

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

##### 29.2 Log file

The same session is **appended** to **`debug/last_build_failure.log`** under the **package root** (via `compiler.WriteBuildFailureLog`, stage `ai_expert`). Structured linker failures from the driver may also append earlier rows for the same run.

##### 29.3 Interactive Expert (TTY only)

If **stdin is a terminal** (detected with `golang.org/x/term`), after the debug block the Expert may:

1. Show a **suggested edit** for high-confidence cases (e.g. `import *` → explicit imports) and prompt **Done? [y/N]** (acknowledgment — you apply the edit in your editor).
2. Open a small **REPL**: type **`why`**, **`zig`**, **`pyqt`**, **`guide`**, **`llm`**, **`strict`**, **`help`**, or **`quit`**.

**Disable interactive mode** (keep the printed debug block): set **`R4D_EXPERT_INTERACTIVE=0`**.

##### 29.4 Implementation rules (for contributors)

- **Regex:** All patterns in `expert.go` use **`regexp.MustCompile`** at package scope.
- **File reads:** Source context uses **`bufio.Scanner`** with an explicit buffer cap (no unbounded slurp of huge files).
- **Panic safety:** The Expert **`defer recover()`** fail-opens to a short raw error line if something unexpected panics.
- **Python vs Roma4D:** The Expert reminds users that **`.py`** is delegated to CPython via **`r4d script.py`**; **`.r4d`** stays on the native pipeline (see CLI help).

##### 29.5 Quick reference for users

| Goal | Command / action |
|------|-------------------|
| Rich failure help | Run without `--strict` (default). |
| CI / scripts only | `r4d --strict file.r4d` |
| Skip post-mortem prompts | `R4D_EXPERT_INTERACTIVE=0` |
| Mirror linker details | `R4D_DEBUG=1` (see §19) |
| Feed an external LLM | Copy from **`LLM_INSTRUCTIONS`** through **`END_LLM_INSTRUCTIONS`** in the debug block |

---

*End of Roma4D Elite Reference Guide.*
