# Pass 9 status — SIMD + GPU auto-offload + incremental compilation

- **Pass:** 9
- **Summary:** Pass 9 completed successfully — SIMD/GPU/incremental with full spacetime support (compile-time regions only; LLVM markers + link stub for GPU path).
- **Files touched this pass:**
  - `roma4d.toml` — `[build] incremental = true`
  - `rt/roma4d_cuda_stub.c` — new CUDA link stub
  - `src/compiler/manifest.go` — parse `incremental` under `[build]`
  - `src/compiler/incremental.go` — new: `PackageInputHash`, `IncrementalCacheDir`
  - `src/compiler/mir.go` — `HasGPUParSpacetime`, par_region `Extra` in string dump
  - `src/compiler/ast_to_mir.go` — `gpu_spacetime_par` when `par` nested under `spacetime:`
  - `src/compiler/codegen_llvm.go` — `<4 x double>` geom_mul; feature globals `simd_geom`, `gpu_par_spacetime`
  - `src/compiler/llvm_link.go` — link CUDA stub when GPU par+spacetime MIR present
  - `examples/hello_4d.roma4d` — `par for` inside `spacetime:`
  - `tests/codegen_test.go`, `tests/spacetime_test.go`, `tests/incremental_test.go` — new assertions
  - `debug/pass9_status.md` — this file
- **Errors / warnings:** None. `go test ./...` from `roma4d/` completed successfully (all packages ok).
- **Timestamp (UTC):** 2026-03-25T18:34:40Z
