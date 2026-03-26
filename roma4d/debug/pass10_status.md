# Pass 10 status — self-hosting & toolchain installation

- **Pass:** 10
- **Summary:** Project complete — Roma4D installed and ready to test (`r4d` / `roma4d` CLI, `go install` from this module).
- **Files created/modified:**
  - `cmd/r4d/main.go` — CLI entry (binary name: `r4d`)
  - `cmd/roma4d/main.go` — alias entry (binary name: `roma4d`)
  - `internal/cli/cli.go` — shared `build`, `run`, `version`, `help`
  - `install.sh`, `install.ps1` — one-shot local `go install` for both commands
  - `roma4d.toml` — final `[package]` description + version `0.1.0`
  - `examples/hello_4d.roma4d` — `r4d run` comment
  - `README.md` — install/run section + layout rows for `cmd/` and `internal/cli`
  - `debug/pass10_status.md` — this file
- **Errors / warnings:** `go test ./...` ok; `go install ./cmd/r4d ./cmd/roma4d` ok. `r4d run` requires **clang** on PATH (fails here if LLVM/clang not installed).
- **Timestamp (UTC):** 2026-03-25T18:40:53Z
