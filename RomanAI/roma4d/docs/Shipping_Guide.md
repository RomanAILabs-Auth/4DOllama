# Shipping Guide — from `.r4d` to shipped native products

Roma4D emits **native executables** (e.g. **`.exe`** on Windows). There is **no** separate runtime VM—what you ship is the **binary** plus any **assets** you bundle yourself.

---

## 1. Produce a release binary

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

## 2. What to ship

| Item | Notes |
|------|--------|
| **Executable** | Single file for many programs; statically linked runtime pieces depend on linker (Zig/clang) and `roma4d_rt.c`. |
| **Data files** | Ship alongside the binary; load via paths you control (no special Roma4D package format required for assets). |
| **License** | Include **LICENSE** from the repo if you redistribute compiler output; respect **third-party** terms of linked runtimes (LLVM, libc, etc.). |

---

## 3. Continuous integration

- **Install:** [Install_Guide.md](Install_Guide.md) — on Linux runners use **`clang`** + **Go**.  
- **Build:** `r4 build --strict your_ci_entry.r4d -o artifact`  
- **Test:** `go test ./...` under **`roma4d/`** validates the **compiler**; your **app** should have its own **`r4 run --strict`** smoke test.

Monorepo **CI** example: see root **`.github/workflows/ci.yml`** (Roma4D + Go modules).

---

## 4. Cross-compilation

Today the **host** toolchain (Zig/clang) defines targets. For exotic targets, align **triple** and linker with your **Zig** / **clang** setup and project policies—consult **Roma4D_Guide §1** (linker) and compiler sources before relying on a specific cross pair in production.

---

## 5. Shipping *services* (4DOllama)

If your “product” is an **API** rather than a CLI tool:

- Build / run **`4dollama serve`** per **[../../docs/4DOllama.md](../../docs/4DOllama.md)**.  
- **Docker:** root **`scripts/install-linux.sh`** + `docker run` pattern.  
- **Ports:** default **13373** (Go) vs **13377** (optional Python bridge)—document for users.

---

## 6. Versioning and reproducibility

- Pin **Go** and **Zig**/**clang** versions in CI (matrix or container image).  
- Commit **`go.sum`** / lockfiles for **reproducible** compiler builds from source.  
- Record **Roma4D** toolchain **version** (`r4d version`) in release notes.

---

## Related

- **Dependencies:** [Dependencies_Guide.md](Dependencies_Guide.md)  
- **Debug failed release builds:** [Debugging_Guide.md](Debugging_Guide.md)  
- **Doc hub:** [README.md](README.md)
