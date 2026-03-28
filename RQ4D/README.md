# RQ4D (RomaQuantum4D)

**RQ4D** — Go **quantum lattice simulator**: complex amplitudes on a 3D grid, Trotter-style steps, optional **tensor-network–style** bond truncation (`--backend=tn`, `--chi`). Mean-field and TN paths are deterministic aside from explicit measurement sampling.

**Repository:** [github.com/RomanAILabs-Auth/RomaQuantum4D](https://github.com/RomanAILabs-Auth/RomaQuantum4D)

**Documentation:** [docs/RQ4D_MASTER_GUIDE.md](docs/RQ4D_MASTER_GUIDE.md) (may still describe the legacy geometric CLI in places; the binary below is the lattice engine.)

## Build, run, install

From the repository root (module `github.com/RomanAILabs-Auth/RomaQuantum4D`):

```bash
go build -o rq4d ./cmd/rq4d          # Unix/macOS
go build -o rq4d.exe ./cmd/rq4d      # Windows
go install ./cmd/rq4d                # installs rq4d to $GOPATH/bin or $GOBIN
```

### Lattice run (default)

```bash
go run ./cmd/rq4d
# or with options:
go run ./cmd/rq4d -lx 8 -ly 8 -lz 8 -steps 30 -backend tn -chi 4 -measure -seed 7
```

Flags include `-lx`, `-ly`, `-lz`, `-dim` (2, 4, or 8), `-dt`, `-steps`, `-j`, `-hz`, `-hx`, `-workers`, `-backend` (`meanfield` | `tn` | `cpu`), `-chi` (1…32 for `tn`), `-measure`, `-collapse`, `-seed`.

### RQ4D-CORE (daemon / bridge)

User-space runtime with a priority scheduler (simulation ahead of external work), optional loopback HTTP **ollama-bridge** API, and `pprof`. No kernel drivers, no process hooking, no Ollama binary changes.

```bash
go build -o rq4d-core ./cmd/rq4d-core
# or equivalent:
go run ./cmd/rq4d core --mode daemon --bridge 127.0.0.1:8744 --tick-ms 100 --backend tn --chi 2
```

Modes: `--mode batch` (default), `--mode daemon`, `--mode interactive`. Daemon flags include `--bridge`, `--ollama-url http://127.0.0.1:11434` (enables optional `POST /v1/ollama/forward`), `--pprof 127.0.0.1:6060`, `--tick-ms`, `--idle-ms`, `--ring-cap`.

### Legacy `.rq4d` script examples

Files under `examples/*.rq4d` targeted the **previous** Cl(4,0) geometric script runner. The current `rq4d` binary does **not** parse those scripts; keep them as reference or remove locally.

### Large-scale demo script

`scripts/RQ4D_World_Record.ps1` was written for the legacy script-based CLI. It is **not** wired to the lattice flags yet; use `go run ./cmd/rq4d` with explicit `-lx/-ly/-lz` for large sweeps (mind memory).

## Roma4D companion

`examples/spacetime_ui_v3.r4d` is **Roma4D** source — run with **`r4d`** from a Roma4D checkout, not `rq4d`.

## Module

`github.com/RomanAILabs-Auth/RomaQuantum4D`

## Copyright & contact

**RomanAILabs** — *Daniel Harding*  
Licensed under **Apache License 2.0** — see [LICENSE](LICENSE) in this directory (or the monorepo root).

**Email:** [romanailabs@gmail.com](mailto:romanailabs@gmail.com) · [daniel@romanailabs.com](mailto:daniel@romanailabs.com)
