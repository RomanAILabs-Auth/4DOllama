// Package coupling implements bidirectional Q-tensor ↔ lattice bridges (user-space, API-only hooks to real tensors).
package coupling

import (
	"math"

	"github.com/4dollama/4dollama/internal/fourd/lattice4"
)

// FrobeniusNorm returns ||A||_F for row-major m×n matrix in data.
func FrobeniusNorm(data []float64, rows, cols int) float64 {
	if len(data) < rows*cols {
		return 0
	}
	var s float64
	for i := 0; i < rows*cols; i++ {
		v := data[i]
		s += v * v
	}
	return math.Sqrt(s)
}

// QKTCognitiveGravity returns ||Q K^T||_F for row-major Q, K (dim×dim). Reuse work buffer of dim*dim to avoid alloc in hot paths.
func QKTCognitiveGravity(Q, K, work []float64, dim int) float64 {
	if dim < 1 || len(Q) < dim*dim || len(K) < dim*dim || len(work) < dim*dim {
		return 0
	}
	// M[i,j] = sum_k Q[i,k]*K[j,k]  (Q * K^T)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var s float64
			for k := 0; k < dim; k++ {
				s += Q[i*dim+k] * K[j*dim+k]
			}
			work[i*dim+j] = s
		}
	}
	return FrobeniusNorm(work, dim, dim)
}

// InjectGravityW adds energy along w-slices proportional to gravity, centered at (cx,cy,cz).
func InjectGravityW(g *lattice4.Grid, gravity float64, cx, cy, cz int, wFalloff float64) {
	if g == nil || gravity == 0 {
		return
	}
	for iw := 0; iw < g.Nw; iw++ {
		dw := float64(iw - g.Nw/2)
		wf := math.Exp(-wFalloff * dw * dw)
		for iz := 0; iz < g.Nz; iz++ {
			for iy := 0; iy < g.Ny; iy++ {
				for ix := 0; ix < g.Nx; ix++ {
					dx := float64(ix - cx)
					dy := float64(iy - cy)
					dz := float64(iz - cz)
					r2 := dx*dx + dy*dy + dz*dz
					rad := math.Exp(-r2 / (2 * float64(max(1, g.Nx/4))))
					g.Add(ix, iy, iz, iw, gravity*wf*rad)
				}
			}
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// LatticeToLogitBias maps mean field energy near w to a small scalar bias for next-token logits (stub).
func LatticeToLogitBias(g *lattice4.Grid, cx, cy, cz, cw int) float64 {
	if g == nil {
		return 0
	}
	v := g.At(cx, cy, cz, cw)
	// compress to [-1,1]-ish
	return math.Tanh(v * 0.01)
}
