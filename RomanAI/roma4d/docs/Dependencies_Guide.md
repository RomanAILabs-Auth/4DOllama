# Dependencies Guide — Roma4D & monorepo tooling

What must be on your machine, what is optional, and which **environment variables** the toolchain respects.

---

## Roma4D compiler (`r4d`, `r4`, `roma4d`)

### Required

| Dependency | Version / notes |
|------------|-----------------|
| **Go** | **1.22+** — builds the driver from `roma4d/cmd/`. `go version` must work. |

### Windows — native linking (pick at least one path)

| Dependency | Role |
|------------|------|
| **Zig** | **Default.** `zig` / `zig.exe` on `PATH`; driver invokes `zig cc`. Override binary with **`R4D_ZIG`** (full path to executable). |
| **LLVM clang + MinGW** | Fallback when Zig is absent. Target **`*-windows-gnu`**. **`R4D_GNU_ROOT`** — MinGW prefix (installer may set from MSYS2 paths). |

You typically do **not** need the MSVC toolchain for Roma4D’s default Windows pipeline.

### Linux

| Dependency | Role |
|------------|------|
| **clang** | Link step; math via `-lm` (handled by driver when linking `rt/roma4d_rt.c`). |
| **build-essential** (or distro equivalent) | C toolchain for the small runtime. |

### macOS

| Dependency | Role |
|------------|------|
| **clang** | Via Xcode CLT or **Homebrew `llvm`**. |
| **Go** | Same as above. |

---

## Environment variables (Roma4D)

| Variable | Purpose |
|----------|---------|
| **`R4D_PKG_ROOT`** | Directory containing **`roma4d.toml`** — set by **Install-R4dUserEnvironment.ps1** (Windows user env). Lets **`r4d /any/path/file.r4d`** resolve imports without `cd` into the clone. |
| **`R4D_ZIG`** | Full path to **Zig** if `zig` is not on `PATH`. |
| **`R4D_GNU_ROOT`** | MinGW/MSYS prefix for **clang** fallback on Windows (headers, `mm_malloc.h`, etc.). |
| **`R4D_DEBUG`** | Non-empty → extra diagnostics (mirrors / supplements `last_build_failure.log` per driver behavior). |

Embedded root: binaries from **`go install -ldflags "-X …EmbeddedPkgRoot=…"`** can bake the repo path so **`R4D_PKG_ROOT`** is optional when using those builds.

---

## Project manifest

| File | Role |
|------|------|
| **`roma4d.toml`** | Package root marker; **`[build] backend = "llvm"`**, edition, etc. The compiler walks **upward** from a `.r4d` file until it finds this file. |

---

## 4DOllama stack (optional, monorepo root)

Not required to **compile Roma4D**. Needed only if you run the **HTTP / GGUF** product.

| Dependency | Role |
|------------|------|
| **Rust + cargo** | Builds **`4d-engine`** (`four_d_engine` staticlib/cdylib). |
| **Go + CGO** | Links **`4dollama`** against `libfour_d_engine` (native path). **`CGO_ENABLED=0`** → stub engine (no full native 4D FFI). |
| **gcc / MSVC** | C compiler for CGO on your platform. |

See **[../../docs/4DOllama.md](../../docs/4DOllama.md)** and root **`scripts/install.ps1`**.

---

## Python `fourdollama` (optional bridge)

Separate from **`4dollama`** (Go). Uses **FastAPI** on **13377** by default. Needs **Python 3.10+** and **`pip install -e` from `4DOllama/`**. Optional **`r4d`** on `PATH` for local kernel runs.

---

## Quick verification

```bash
go version
# Windows:
zig version
# or: clang --version
# Linux/macOS:
clang --version
r4d version
```

**Install:** [Install_Guide.md](Install_Guide.md) · **Ship binaries:** [Shipping_Guide.md](Shipping_Guide.md)
