package quantum

import "math"

// StateLayout: flat arrays Re, Im with length N*Dim (node-major).
// Site i, local component a (0..Dim-1): index = i*Dim + a.
// Dim must be a power of two: local Hilbert space (C^2)^{⊗ nq}, Dim = 2^nq.

// Simulator holds lattice quantum state (double buffer), topology, and evolution parameters.
type Simulator struct {
	LatticeTopo
	Dim    int // local Hilbert dimension d ∈ {2,4,8}
	NQ     int // log2(Dim); number of qubits per site
	N      int // node count (cached)
	ReA, ImA []float64
	ReB, ImB []float64
	Workers  int

	// Hamiltonian parameters (used in evolve.go / energy.go).
	Dt    float64 // Δt
	JBond float64 // XX coupling strength between paired qubits on an edge
	Hz    []float64
	Hx    []float64
}

// NewSimulator allocates state for an Lx×Ly×Lz torus with local dimension Dim (2, 4, or 8).
func NewSimulator(lx, ly, lz, dim int, workers int) *Simulator {
	if dim != 2 && dim != 4 && dim != 8 {
		panic("quantum.NewSimulator: Dim must be 2, 4, or 8")
	}
	nq := 0
	switch dim {
	case 2:
		nq = 1
	case 4:
		nq = 2
	case 8:
		nq = 3
	}
	n := lx * ly * lz
	if workers <= 0 {
		workers = 1
	}
	return &Simulator{
		LatticeTopo: LatticeTopo{Lx: lx, Ly: ly, Lz: lz},
		Dim:         dim,
		NQ:          nq,
		N:           n,
		ReA:         make([]float64, n*dim),
		ImA:         make([]float64, n*dim),
		ReB:         make([]float64, n*dim),
		ImB:         make([]float64, n*dim),
		Workers:     workers,
		Hz:          make([]float64, nq),
		Hx:          make([]float64, nq),
	}
}

// SiteOffset returns the flat index of site i's component 0.
func (S *Simulator) SiteOffset(site int) int {
	return site * S.Dim
}

// GlobalNorm2 returns Σ_{i,a} |Re+iIm|^2 (should be 1 after normalized init).
func (S *Simulator) GlobalNorm2(re, im []float64) float64 {
	var s float64
	for i := 0; i < S.N*S.Dim; i++ {
		s += re[i]*re[i] + im[i]*im[i]
	}
	return s
}

// NormalizeGlobal scales re, im so GlobalNorm2 = 1.
func (S *Simulator) NormalizeGlobal(re, im []float64) {
	n2 := S.GlobalNorm2(re, im)
	if n2 <= 0 || math.IsNaN(n2) {
		return
	}
	inv := 1.0 / math.Sqrt(n2)
	for i := 0; i < S.N*S.Dim; i++ {
		re[i] *= inv
		im[i] *= inv
	}
}

// InitProductComputational sets every site to |0…0⟩ (component 0 amplitude 1).
func (S *Simulator) InitProductComputational() {
	clear(S.ReA)
	clear(S.ImA)
	for i := 0; i < S.N; i++ {
		o := S.SiteOffset(i)
		S.ReA[o] = 1
	}
	clear(S.ReB)
	clear(S.ImB)
}
