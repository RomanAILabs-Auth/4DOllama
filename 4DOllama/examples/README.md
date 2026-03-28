# 4DOllama examples

## Native engine (Go, monorepo root)

Run from **`4DEngine/`** (parent of this folder), where `go.mod` defines `github.com/4dollama/4dollama`:

```powershell
go run ./cmd/4dollama fourd ga-demo
go run ./cmd/4dollama fourd lattice -steps 80
```

## RQ4D lattice scripts (separate module)

If you use **[RomaQuantum4D](https://github.com/RomanAILabs-Auth/RomaQuantum4D)** (`RQ4D/`), the **`.rq4d`** ISA and **`rq4d core`** daemon are documented in `RQ4D/docs/RQ4D_MASTER_GUIDE.md`.  
Files named `*.rq4d` in this folder are **placeholders or cross-links** unless copied next to an RQ4D `go.mod`.

## Roma4D language (`.r4d`)

Geometric kernels and `par for` worldtubes belong in **`roma4d/examples/`** and compile with **`r4d`**, not Python.

## Contact

**RomanAILabs** — *Daniel Harding* — [romanailabs@gmail.com](mailto:romanailabs@gmail.com) · [daniel@romanailabs.com](mailto:daniel@romanailabs.com)
