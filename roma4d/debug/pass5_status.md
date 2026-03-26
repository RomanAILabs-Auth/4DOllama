# Roma4D — Pass 5 (Mid-level IR, Systems Edition) status

## Pass number

**Pass 5 — Mid-level IR (MIR): SSA-oriented ops, SoA load/store, `par` / `unsafe` regions, raw pointers, ownership metadata on defs.**

## Files created (this pass)

- `src/compiler/mir.go` — MIR types, instructions, module/function dump (`String()`)
- `src/compiler/ast_to_mir.go` — `LowerToMIR(*CheckResult)`, AST walk + type/ownership hints
- `tests/mir_test.go` — lowers `examples/hello_4d.roma4d`, asserts MIR contains key markers; rejects lowering when check fails
- `debug/pass5_status.md` — this file

## Files modified (this pass)

- `src/parser/token.go` — `KwUnsafe`, keyword `unsafe`, `IsKeyword` range
- `src/parser/ast.go` — `UnsafeStmt`
- `src/parser/parser.go` — compound start `KwUnsafe`
- `src/parser/parser_compound.go` — `parseUnsafeStmt`, `parseCompoundStmt` dispatch
- `src/compiler/types.go` — `RawPtr`, `TypRawPtr`, `TypeIsSendable` case
- `src/compiler/typechecker.go` — `UnsafeStmt` checking; `rawptr` annotation; `mir_alloc` / `mir_ptr_load` / `mir_ptr_store`; `seedBuiltins` Sendable from callable result
- `src/compiler/ownership.go` — `UnsafeStmt` borrow frame; par-capture / taint name walks for `unsafe` bodies
- `examples/hello_4d.roma4d` — `unsafe:` block with manual alloc + ptr load/store

## Confirmed unchanged / not modified

- `roma4d.toml` — `[systems] gc = false` and `unsafe = true` already present (verified)
- `src/core/4d/` — not touched

## Status summary

**Pass 5 completed successfully** — `go test ./...` from `roma4d/` passes.

## Errors / warnings

- MIR v0 elides full list comprehension and list literal bodies (comment + placeholder const); Pass 6 should expand.
- Single-block functions only; CFG splits deferred to codegen.
- `par_region` / `unsafe_region` nest flat instruction lists, not separate basic blocks yet.

## Timestamp

2026-03-25 (local, after `go test ./...` green)
