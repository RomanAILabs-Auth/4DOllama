// Package quantum is a discrete quantum lattice simulator (complex amplitudes, unitary Trotter
// evolution, optional projective measurement). See state.go and evolve.go for dynamics.
package quantum

// LatticeTopo is a 3D periodic grid with 6-nearest-neighbor (face) adjacency.
type LatticeTopo struct {
	Lx, Ly, Lz int
}

// NodeCount returns Lx * Ly * Lz.
func (T *LatticeTopo) NodeCount() int {
	return T.Lx * T.Ly * T.Lz
}

// IdxNode maps periodic coordinates to flat node index in [0, NodeCount).
func (T *LatticeTopo) IdxNode(ix, iy, iz int) int {
	ix = ((ix % T.Lx) + T.Lx) % T.Lx
	iy = ((iy % T.Ly) + T.Ly) % T.Ly
	iz = ((iz % T.Lz) + T.Lz) % T.Lz
	return (iz*T.Ly+iy)*T.Lx + ix
}

// FlatToGrid decodes flat node index to (ix, iy, iz).
func (T *LatticeTopo) FlatToGrid(flat int) (ix, iy, iz int) {
	lx, ly := T.Lx, T.Ly
	iz = flat / (lx * ly)
	rem := flat - iz*lx*ly
	iy = rem / lx
	ix = rem % lx
	return ix, iy, iz
}
