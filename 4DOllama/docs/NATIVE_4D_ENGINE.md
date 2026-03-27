# 4DOllama native 4D engine — architecture & phased roadmap

**Module:** `github.com/4dollama/4dollama` (monorepo root)  
**Primary language (kernels):** Roma4D **`.r4d`** — ahead-of-time compiled geometric programs.  
**Execution / orchestration:** Go runtime in this repo (`4dollama` CLI, HTTP surface, **fourd** numerical core).

This document states **mathematical intent** and **what is implemented today** without conflating vision with shipped code.

---

## Mathematical foundation (normative for design)

| Concept | Definition |
|--------|------------|
| **Point** | Grade-1 multivector \(P = x e_1 + y e_2 + z e_3 + w e_4\) in **Cl(4,0)** (\(e_i^2=1\), \(e_ie_j=-e_je_i\)). |
| **Rotor** | \(R = \exp(-\frac{\theta}{2} B)\) with unit bivector \(B\); sandwich \(P' = R P \tilde R\). |
| **Isoclinic motion** | Two rotations in orthogonal planes (e.g. \(e_1e_2\) and \(e_3e_4\)) at equal angle — implemented as **composed rotors** in `internal/fourd/clifford`. |
| **4+1 model** | \(w\) is a **spatial** fourth axis; **simulation time** is the engine step index (external to \((x,y,z,w)\)). |
| **Lattice PDE** | Discrete **Laplacian** on a 4-torus \(\nabla^2 \phi\) with neighbors ±1 per axis; **heat / diffusion** step is **stable** for small \(D\Delta t\). **Leapfrog wave** exists (`lattice4.WaveState`) but requires **CFL tuning** for production. |
| **Q-tensor coupling** | \(\|Q K^\top\|_F\) (Frobenius) as **cognitive gravity** proxy → **source field** on the lattice; reverse path: **logit bias** stub from local \(\phi\) (`coupling.LatticeToLogitBias`). |
| **Hodge / harmonic** | Full **discrete Hodge decomposition** and **Betti numbers** on a 4D cubical complex are **Phase 3** research. **Shipped:** mean removal (constant mode) + helpers for future DEC. |

---

## Phase status (repository truth)

| Phase | Scope | Status |
|-------|--------|--------|
| **1** | Cl(4,0) product, rotors, isoclinic demo; 4D grid, Laplacian, heat step, leapfrog wave | **Partially shipped** (`internal/fourd/...`) |
| **2** | Live attention tensors ↔ lattice | **API + mock tensors** in orchestrator; wire to real GGUF/Ollama tensors in `internal/inference` / HTTP layer (future) |
| **3** | Exact discrete Hodge, Betti | **Documented + stubs** (`internal/fourd/hodge`) |
| **4** | 3D projection, WASD viewer | **Not shipped** — recommend external tool (ParaView, custom OpenGL) consuming exported grids |
| **5** | Master orchestrator, auto experiments | **Minimal** `RunLatticeLoop`; extend with job specs + logging |
| **6** | Polish, lock-free hot paths, exports | **Incremental**; graph executor lives under `RQ4D/internal/core` for macro-DAG workloads |

---

## Commands

From monorepo root (where `go.mod` is):

```bash
go build -o 4dollama ./cmd/4dollama
./4dollama fourd ga-demo
./4dollama fourd lattice -steps 100 -kappa 0.002 -inject-every 8
```

---

## Honesty clause

- **4dollama** remains **Ollama-compatible** for model pull/chat; the **fourd** path is the **native 4D numerical substrate**, not a replacement for the whole inference stack in one release.
- **“All operations native GA”** for every token of inference is a **roadmap** item: numerical hot paths today mix **Go lattice** + **Cl(4,0) library**; Roma4D **`.r4d`** should own rotor-heavy kernels as they are ported from reference Python.
