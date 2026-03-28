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

	// BackendMeanField (default) or BackendTN; BackendCPU aliases meanfield.
	Backend string
	Chi     int // bond dimension χ (1 = product / mean-field path)

	// TN-only: reduced density matrices ρ per site (real then imag blocks, d×d each), double-buffered.
	RhoA, RhoB []float64
	Bonds      []BondState // length 3N: +x,+y,+z edges per cell index

	// Set during Step() for TN multi-bond evolution (not used concurrently).
	rhoStepRead, rhoStepWrite []float64

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
		Backend:     BackendMeanField,
		Chi:         1,
		Hz:          make([]float64, nq),
		Hx:          make([]float64, nq),
	}
}

// SetQuantumBackend selects mean-field or tensor-network evolution and optional χ.
// Re-allocates rho/bond storage when switching to BackendTN.
func (S *Simulator) SetQuantumBackend(backend string, chi int) {
	if backend == "" || backend == BackendCPU {
		backend = BackendMeanField
	}
	S.Backend = backend
	if chi < 1 {
		chi = 1
	}
	if chi > MaxChiCap {
		chi = MaxChiCap
	}
	S.Chi = chi
	if S.Backend == BackendTN {
		S.initTNStorage()
	} else {
		S.RhoA, S.RhoB, S.Bonds = nil, nil, nil
	}
}

// GlobalTraceRho returns Σ_s Tr(ρ_s) when TN buffers exist (each Tr(ρ_s) should be 1).
func (S *Simulator) GlobalTraceRho() float64 {
	if S.RhoA == nil {
		return 0
	}
	d := S.Dim
	var t float64
	for s := 0; s < S.N; s++ {
		rre, _ := rhoPtrFlat(S.RhoA, s, d)
		for a := 0; a < d; a++ {
			t += rre[a*d+a]
		}
	}
	return t
}

// SyncAllRhoFromPsi sets ρ = |ψ⟩⟨ψ| / Tr from amplitudes in re, im (one buffer pair).
func (S *Simulator) SyncAllRhoFromPsi(re, im []float64) {
	if S.RhoA == nil {
		return
	}
	d := S.Dim
	for s := 0; s < S.N; s++ {
		syncSiteRhoFromPsi(re, im, S.RhoA, s, d)
	}
}

func syncSiteRhoFromPsi(re, im []float64, rhoFlat []float64, site, d int) {
	br, bi := rhoPtrFlat(rhoFlat, site, d)
	base := site * d
	for a := 0; a < d; a++ {
		ar, ai := re[base+a], im[base+a]
		for b := 0; b < d; b++ {
			br_, bi_ := re[base+b], im[base+b]
			br[a*d+b] = ar*br_ + ai*bi_
			bi[a*d+b] = ai*br_ - ar*bi_
		}
	}
	var tr float64
	for a := 0; a < d; a++ {
		tr += br[a*d+a]
	}
	if tr > 1e-30 {
		inv := 1.0 / tr
		for i := 0; i < d*d; i++ {
			br[i] *= inv
			bi[i] *= inv
		}
	}
}

// PsiFromRhoPrimary writes each site's dominant eigenvector of ρ (in RhoA) into ReA/ImA.
func (S *Simulator) PsiFromRhoPrimary() {
	if S.RhoA == nil {
		return
	}
	d := S.Dim
	for s := 0; s < S.N; s++ {
		psiFromRhoIntoSite(S.RhoA, s, d, S.ReA, S.ImA)
	}
}

func psiFromRhoIntoSite(rhoFlat []float64, site, d int, outRe, outIm []float64) {
	br, bi := rhoPtrFlat(rhoFlat, site, d)
	base := site * d
	var vRe, vIm [maxDim]float64
	var wRe, wIm [maxDim]float64
	for i := 0; i < d; i++ {
		vRe[i] = math.Sin(float64(i+site)*0.31) * 0.2
		vIm[i] = math.Cos(float64(i+site)*0.47) * 0.2
	}
	normV(vRe[:], vIm[:], d)
	for iter := 0; iter < 40; iter++ {
		hermMatVec(br, bi, d, vRe[:], vIm[:], wRe[:], wIm[:])
		copy(vRe[:d], wRe[:d])
		copy(vIm[:d], wIm[:d])
		normV(vRe[:], vIm[:], d)
	}
	for a := 0; a < d; a++ {
		outRe[base+a] = vRe[a]
		outIm[base+a] = vIm[a]
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
	if S.Backend == BackendTN && S.RhoA != nil {
		S.InitRhoComputational()
		copy(S.RhoB, S.RhoA)
	}
}
