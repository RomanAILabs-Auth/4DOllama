// Package hodge provides discrete Hodge-style projections on a 4D periodic scalar lattice.
// Full DEC on a 4-manifold with exact Betti tracking is future work; here we ship stable
// *harmonic-ish* extraction via mean removal + Laplacian smoothing (proxy for low-mode projection).
package hodge

import (
	"github.com/4dollama/4dollama/internal/fourd/lattice4"
)

// RemoveMean sets phi := phi - mean(phi) on the torus (exact constant harmonic complement for 0-forms).
func RemoveMean(g *lattice4.Grid) {
	if g == nil || len(g.Data) == 0 {
		return
	}
	var s float64
	for _, v := range g.Data {
		s += v
	}
	m := s / float64(len(g.Data))
	for i := range g.Data {
		g.Data[i] -= m
	}
}

// SmoothLaplacianStep applies one explicit Jacobi-like smoothing: phi -= alpha * Lap(phi) (stability: small alpha).
func SmoothLaplacianStep(dst, src *lattice4.Grid, alpha float64) {
	if dst == nil || src == nil || len(dst.Data) != len(src.Data) {
		return
	}
	nx, ny, nz, nw := src.Nx, src.Ny, src.Nz, src.Nw
	for iw := 0; iw < nw; iw++ {
		for iz := 0; iz < nz; iz++ {
			for iy := 0; iy < ny; iy++ {
				for ix := 0; ix < nx; ix++ {
					lap := src.Laplacian4(ix, iy, iz, iw)
					v := src.At(ix, iy, iz, iw)
					dst.Set(ix, iy, iz, iw, v-alpha*lap)
				}
			}
		}
	}
}

// HarmonicProxy runs mean removal + k smoothing iterations (not exact Hodge decomposition).
func HarmonicProxy(g *lattice4.Grid, scratch *lattice4.Grid, iterations int, alpha float64) {
	if g == nil {
		return
	}
	RemoveMean(g)
	if scratch == nil || iterations <= 0 {
		return
	}
	a, b := g, scratch
	for i := 0; i < iterations; i++ {
		SmoothLaplacianStep(b, a, alpha)
		a, b = b, a
	}
	if a != g {
		copy(g.Data, a.Data)
	}
}

// EnergyDensity returns phi^2 at each site (for visualization export).
func EnergyDensity(phi *lattice4.Grid, out *lattice4.Grid) {
	if phi == nil || out == nil {
		return
	}
	for i := range phi.Data {
		out.Data[i] = phi.Data[i] * phi.Data[i]
	}
}

// DualPlanePair documents the orthogonal 2-plane split (xy) ↔ (zw) in 4D Euclidean indexing.
// Semantic mapping into GA planes is done in Roma4D sources; this is the lattice bookkeeping convention.
func DualPlanePair() (planeXY, planeZW string) {
	return "e1e2", "e3e4"
}
