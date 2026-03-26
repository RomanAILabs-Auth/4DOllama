# Pass 7 status — Spacetime Extension Sprint #1 (Option B)

**Pass:** 7 (spacetime foundation: types, ownership regions, MIR, LLVM markers)

**Status:** Pass 7 completed successfully – spacetime foundation added (`go test ./...` from `roma4d/` passes).

**Files created or modified in this pass only:**

| Path | Role |
|------|------|
| `src/parser/token.go` | `KwTime`, `KwSpacetime`, keyword table, `IsKeyword` range |
| `src/parser/ast.go` | `TimeCoord`, `SpacetimeStmt`; `TimeCoord.Typ` for checker |
| `src/parser/parser.go` | `isCompoundStart` / `isStmtStart`; `tokString` uses `IsKeyword()` |
| `src/parser/parser_compound.go` | `parseSpacetimeStmt` |
| `src/parser/parser_expr.go` | Parse `t` as `TimeCoord` |
| `src/compiler/types.go` | `TimeDim`, `SpacetimeType`, `TypTime`, `StripSpacetime`, `TypeIsSendable`, `isGA` |
| `src/compiler/mir.go` | `MIRTypeRef.TimeTag`; `MIRTemporalMove`, `MIRSpacetimeRegion`, `MIRTimeTravelBorrow`; formatting |
| `src/compiler/typechecker.go` | `time` annotation; `TimeCoord`; `@` → `SpacetimeType`; `timetravel_borrow`; `assignable` / `checkAnnAssign` / `checkStmt` spacetime; `typeBinOp` signature; GA helpers strip spacetime |
| `src/compiler/ownership.go` | `tEpoch` on borrow frames; inherit vs `spacetime:` epoch; `timetravel_borrow` / par-capture like `borrow` |
| `src/compiler/ast_to_mir.go` | Lower `TimeCoord`, `expr @ t`, `spacetime:`, `timetravel_borrow`; `borrow`/`mutborrow` Copy uses SSA operand (LLVM fix) |
| `src/compiler/codegen_llvm.go` | Feature globals `has_spacetime_region`, `has_temporal`, `has_timetravel_borrow`; lower temporal MIR; `mirTypeRefToLLVM` strips `@t` suffix |
| `examples/hello_4d.roma4d` | Demo: `time`, `pos[0] @ t`, `tmp_st`, `spacetime:` + `timetravel_borrow` |
| `tests/spacetime_test.go` | Parse + MIR substring tests |
| `tests/codegen_test.go` | Assert new LLVM marker globals |
| `debug/pass7_status.md` | This file |

**Errors / warnings encountered:**

- **`t` as keyword** shadows Python-style identifier `t` (intentional for Option B).
- **Discard `_` reuse** in `hello_4d` caused bogus type clashes (`tuple` vs `vec4`, `Borrowed[rotor]` vs `linear[vec4]`); fixed with distinct names (`tmp_st`, `again`, `_hold`).
- **`borrow`/`mutborrow`/`timetravel_borrow` MIR Copy** had empty `Uses`, breaking LLVM lowering; fixed to copy the borrowed SSA value.
- **LLVM:** `temporal_move` is a **warning** (identity lowering) until Pass 8+ adds real temporal ops.

**Timestamp:** 2026-03-25 18:05:03 -07:00 — verified with `go test ./...`.
