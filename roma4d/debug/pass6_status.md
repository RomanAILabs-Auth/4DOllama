# Pass 6 status — LLVM codegen (CPU)

**Pass:** 6 (MIR → LLVM IR, clang link)

**Status:** Pass 6 completed successfully (`go test ./...` from `roma4d/` passes).

**Files created or modified in this pass (Pass 6 scope):**

| Path | Role |
|------|------|
| `go.mod`, `go.sum` | Dependency `github.com/llir/llvm` |
| `src/compiler/codegen_llvm.go` | `LowerMIRToLLVM`, MIR lowering, SoA/par/unsafe/geom stubs, slot typing |
| `src/compiler/llvm_link.go` | `MIRToLLVMIR`, `FindClang`, `BuildExecutable` |
| `rt/roma4d_rt.c` | Runtime stubs for link (bump, GA, list get, etc.) |
| `examples/min_main.roma4d` | Minimal `main` → native smoke (exit 42) |
| `examples/hello_4d.roma4d` | Adjusted as needed for pipeline / markers |
| `tests/codegen_test.go` | LLVM IR string checks + optional `BuildExecutable` test |
| `roma4d.toml` | `[build] backend = "llvm"` |
| `debug/pass6_status.md` | This file |

**Errors / warnings encountered (resolved):**

- **llir API:** Corrected `types.NewArray` order, `Param.Typ`, `FuncType.RetType`, `CharArray.Type()` for GEP.
- **Store type panic (`i8*` → `i64*`):** Same source local name could be stored with different MIR types while the first `alloca` stayed `i64`. **Fix:** track `slotTypes` per name; on type change, allocate a fresh stack slot and warn (`local %q: slot type changed …`).
- **Coercion:** Extended `coerceForStore` for `i32`→`i64`, and pointer→`i8*` via `bitcast` for SoA/array pointers.

**Notes:** `TestBuildExecutableMinMain` skips if `clang` is not on `PATH`. Full `hello_4d` native run may need richer `roma4d_rt.c` symbols later.

**Timestamp:** 2026-03-25 17:46:14 -07:00
