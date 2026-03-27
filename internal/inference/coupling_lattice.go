// Package inference: coupling_lattice.go wires the 4D scalar lattice to autoregressive decode.
// Each token step: pack RoPE + spacetime-attention quads into Q,K-shaped buffers,
// compute ||QK^T||_F as cognitive-gravity proxy, inject as PDE source, diffuse one step,
// then read reverse coupling as a logit bias (see coupling.LatticeToLogitBias).
package inference

import (
	"math"
	"os"
	"strconv"
	"sync"

	"github.com/4dollama/4dollama/internal/fourd/coupling"
	"github.com/4dollama/4dollama/internal/fourd/lattice4"
)

const inferLatticeDim = 16 // Q,K are 16×16 for Frobenius coupling (macro chunk, not per-parameter)

// InferenceLattice couples token-stream attention geometry to a 4-torus field (w-axis aware injection).
type InferenceLattice struct {
	mu    sync.Mutex
	cur   *lattice4.Grid
	nxt   *lattice4.Grid
	src   *lattice4.Grid
	qFlat []float64
	kFlat []float64
	work  []float64
	kappa float64
	D     float64
	dt    float64
}

var (
	globalLattice     *InferenceLattice
	globalLatticeOnce sync.Once
)

// GlobalLattice returns the process-wide coupling field (one lattice per 4dollama process).
func GlobalLattice() *InferenceLattice {
	globalLatticeOnce.Do(func() {
		globalLattice = NewInferenceLattice()
	})
	return globalLattice
}

// NewInferenceLattice builds a small periodic 4D grid suitable for hot-loop updates (no heap in OnTokenStep after init).
func NewInferenceLattice() *InferenceLattice {
	kappa := 0.0015
	if s := os.Getenv("FOURD_LATTICE_KAPPA"); s != "" {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			kappa = v
		}
	}
	nx, ny, nz, nw := 10, 10, 10, 6
	return &InferenceLattice{
		cur:   lattice4.NewGrid(nx, ny, nz, nw),
		nxt:   lattice4.NewGrid(nx, ny, nz, nw),
		src:   lattice4.NewGrid(nx, ny, nz, nw),
		qFlat: make([]float64, inferLatticeDim*inferLatticeDim),
		kFlat: make([]float64, inferLatticeDim*inferLatticeDim),
		work:  make([]float64, inferLatticeDim*inferLatticeDim),
		kappa: kappa,
		D:     0.014,
		dt:    0.18,
	}
}

// packRK fills dst (row-major dim×dim) from two float32 streams (RoPE and attention output).
func packRK(dst []float64, rope, attn []float32, dim int) {
	n := dim * dim
	la := len(rope)
	lb := len(attn)
	if la == 0 && lb == 0 {
		return
	}
	for i := 0; i < n; i++ {
		var va, vb float32
		if la > 0 {
			va = rope[i%la]
		}
		if lb > 0 {
			vb = attn[i%lb]
		}
		// scale to O(1) Frobenius without blowing up matmul
		dst[i] = float64(va)*3e-5 + float64(vb)*3e-5
	}
}

// OnTokenStep runs one bidirectional coupling tick: QK gravity → heat step → lattice → logit bias.
func (il *InferenceLattice) OnTokenStep(rope, attn, lifted []float32, step int) float64 {
	if il == nil {
		return 0
	}
	il.mu.Lock()
	defer il.mu.Unlock()

	packRK(il.qFlat, rope, attn, inferLatticeDim)
	packRK(il.kFlat, attn, rope, inferLatticeDim)
	g := coupling.QKTCognitiveGravity(il.qFlat, il.kFlat, il.work, inferLatticeDim) * il.kappa
	if g > 1e6 {
		g = 1e6
	}
	cx, cy, cz := il.cur.Nx/2, il.cur.Ny/2, il.cur.Nz/2
	coupling.InjectGravityW(il.src, g, cx, cy, cz, 0.14+0.01*float64(step%7))
	lattice4.StepHeat(il.cur, il.nxt, il.src, il.D, il.dt)
	il.cur, il.nxt = il.nxt, il.cur
	clear(il.src.Data)
	cw := il.cur.Nw / 2
	if cw < 0 {
		cw = 0
	}
	b := coupling.LatticeToLogitBias(il.cur, cx, cy, cz, cw)
	// strengthen coupling slightly with step depth (bounded)
	return b * (1.0 + 0.02*math.Sqrt(float64(step+1)))
}

// ApplyLogitBias adds lattice-derived bias to logits (in-place, softmax-stable shift).
func ApplyLogitBias(logits []float32, bias float64) {
	if len(logits) == 0 || bias == 0 {
		return
	}
	delta := float32(bias * 2.0)
	for i := range logits {
		logits[i] += delta
	}
}
