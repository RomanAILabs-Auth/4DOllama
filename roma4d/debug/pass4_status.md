# Roma4D — Pass 4 (Fixed, Systems Edition) status

## Pass number

**Pass 4 — Ownership 2.0 checking (fixed): block-scoped borrows, linear preservation, discard `_`, `[systems]` manifest.**

## Files created

- `debug/pass4_status.md` (this file)
- `debug/` directory

## Files modified

- `src/compiler/ownership.go` — block-scoped borrow stack (`pushBorrowFrame` / `popBorrowFrame`); borrows end at block exit; `endStmt` no longer releases borrows
- `src/compiler/typechecker.go` — `storedTypeAfterAssign`, `checkAnnAssign` / `assignTarget` preserve `Linear{T}`; `assignable` widened for linear slots; discard `_` metadata on symbols
- `src/compiler/symbol.go` — `DiscardName`, `Discard` field on `Symbol`
- `src/compiler/scope.go` — `Define` allows repeated `_`
- `src/compiler/manifest.go` — optional `[systems]` keys `gc`, `unsafe` (boolean parsing)
- `roma4d.toml` — `[systems] gc = false`, `unsafe = true`
- `examples/hello_4d.roma4d` — header notes block-scoped borrow; linear SoA + multiple `_` (discard) assignments

## Status summary

**Pass 4 fixed successfully** — `go test ./...` from `roma4d/` passes (parser, tests/ownership, tests/typechecker, core/4d).

## Remaining notes / warnings

- `SystemsGCEnabled` defaults to `false` when `[systems]` is absent (zero value); packages that want GC should set `gc = true` explicitly once backends consume the manifest.
- SoA slot keys are still per `(base symbol, field)` without alias analysis; stricter flow sensitivity can land in a later pass.

## Timestamp

2026-03-25 17:09:01 -07:00
