# Install Guide — 4DEngine / Roma4D

**Audience:** Anyone cloning this monorepo on **Linux, macOS, or Windows**.

This guide installs **Roma4D** (`r4d`, `r4`, `roma4d`) so you can compile and run `.r4d` files.  
Optional: **4DOllama** (Go `4dollama` + Rust `four_d_engine`) is a **separate** step at the end.

**Path convention:** Commands below assume your **current directory is the repository root** — the folder that contains **`roma4d/`**, **`go.mod`** (root), and **`scripts/`**. If you are inside `roma4d/`, `cd ..` first.

**More docs:** See **[Documentation hub (README)](README.md)** for use, LLM workflow, debugging, errors, dependencies, and shipping.

---

## 0. Clone the repository

```bash
git clone https://github.com/RomanAILabs-Auth/4DOllama.git 4DEngine
cd 4DEngine
```

(If you use another remote or folder name, keep using that path below.)

---

## 1. Install prerequisites

### All platforms

| Need | Why |
|------|-----|
| **[Go 1.22+](https://go.dev/dl/)** | Builds the Roma4D compiler. Must be on `PATH` (`go version` works). |

### Windows only

| Need | Why |
|------|-----|
| **[Zig](https://ziglang.org/download/)** on `PATH` | Default linker (`zig cc`). Install e.g. `winget install Zig.Zig` and **open a new terminal**. |
| **OR** LLVM **clang** + **MinGW-w64** (e.g. [MSYS2](https://www.msys2.org/)) | Fallback if you do not use Zig. The installer tries to set `R4D_GNU_ROOT` when MSYS is in the usual location. |

You do **not** need Visual Studio for the default Roma4D Windows build (toolchain targets `*-windows-gnu`).

### Linux

| Need | Why |
|------|-----|
| **`clang`** | Links native binaries. e.g. `sudo apt install clang build-essential` (Debian/Ubuntu). |
| **`git`** | Clone / modules. |

### macOS

| Need | Why |
|------|-----|
| **Xcode Command Line Tools** or **Homebrew `llvm`** | So `clang` is available. e.g. `brew install go llvm`. |

**Deep dive:** [Dependencies_Guide.md](Dependencies_Guide.md)

---

## 2. One-command install (Roma4D)

Run from the **repository root**.

### Windows (PowerShell)

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

### Linux and macOS (bash/sh)

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

## 3. Verify Roma4D

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

## 4. If something fails (checklist)

1. **`go: command not found`** — Install Go; open a **new** terminal.
2. **Windows: scripts disabled** — Use `-ExecutionPolicy Bypass` (§2) or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. **`r4d` not found** — New terminal (Windows); add `GOPATH/bin` to `PATH` (Unix).
4. **Link errors (Windows)** — Install **Zig**, or **clang + MinGW**, then re-run the installer.
5. **Build log** — `roma4d/debug/last_build_failure.log`; set **`R4D_DEBUG=1`**. See [Debugging_Guide.md](Debugging_Guide.md).
6. **Wrong binary** — `where r4d` (Windows) / `which r4d` (Unix).

**More:** [Errors_Guide.md](Errors_Guide.md)

---

## 5. Manual install (without universal scripts)

From **repo root**:

```bash
cd roma4d
```

**Windows:** `.\Install-Full.ps1`  
**Linux / macOS:** `chmod +x install-full.sh && ./install-full.sh`

---

## 6. Optional — 4DOllama (same monorepo, different stack)

**4DOllama:** Ollama-shaped HTTP API + **`4dollama`** Go CLI (default port **13373**). Needs **Rust** + **Go CGO** on Windows for full native engine (stub builds possible without CGO).

- **Windows (repo root):**  
  `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1`
- **Docs:** [../../docs/4DOllama.md](../../docs/4DOllama.md) (from this file: monorepo `docs/` folder).
- **Linux Docker:** `./scripts/install-linux.sh`

Do **not** confuse **`4dollama`** (Go) with **`fourdollama`** (optional Python bridge, port **13377**).

---

## 7. Where things live (repo layout)

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
