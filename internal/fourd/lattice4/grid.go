// Package lattice4 implements a periodic 4D scalar lattice for wave PDEs and coupling fields.
package lattice4

// Grid is a 4-torus of scalar samples (phi, energy, etc.).
type Grid struct {
	Nx, Ny, Nz, Nw int
	Data           []float64
}

// NewGrid allocates zeroed data of size Nx*Ny*Nz*Nw.
func NewGrid(nx, ny, nz, nw int) *Grid {
	if nx < 2 || ny < 2 || nz < 2 || nw < 2 {
		nx, ny, nz, nw = 4, 4, 4, 4
	}
	n := nx * ny * nz * nw
	return &Grid{Nx: nx, Ny: ny, Nz: nz, Nw: nw, Data: make([]float64, n)}
}

func (g *Grid) idx(ix, iy, iz, iw int) int {
	ix = ((ix % g.Nx) + g.Nx) % g.Nx
	iy = ((iy % g.Ny) + g.Ny) % g.Ny
	iz = ((iz % g.Nz) + g.Nz) % g.Nz
	iw = ((iw % g.Nw) + g.Nw) % g.Nw
	return ((iw*g.Nz+iz)*g.Ny+iy)*g.Nx + ix
}

// At returns value at periodic coordinates.
func (g *Grid) At(ix, iy, iz, iw int) float64 {
	return g.Data[g.idx(ix, iy, iz, iw)]
}

// Set sets value.
func (g *Grid) Set(ix, iy, iz, iw int, v float64) {
	g.Data[g.idx(ix, iy, iz, iw)] = v
}

// Add adds delta (e.g. cognitive gravity injection).
func (g *Grid) Add(ix, iy, iz, iw int, v float64) {
	i := g.idx(ix, iy, iz, iw)
	g.Data[i] += v
}

// Laplacian4 returns discrete Laplacian (second central differences, periodic) at one site.
func (g *Grid) Laplacian4(ix, iy, iz, iw int) float64 {
	c := g.At(ix, iy, iz, iw)
	lx := g.At(ix-1, iy, iz, iw) + g.At(ix+1, iy, iz, iw) - 2*c
	ly := g.At(ix, iy-1, iz, iw) + g.At(ix, iy+1, iz, iw) - 2*c
	lz := g.At(ix, iy, iz-1, iw) + g.At(ix, iy, iz+1, iw) - 2*c
	lw := g.At(ix, iy, iz, iw-1) + g.At(ix, iy, iz, iw+1) - 2*c
	return lx + ly + lz + lw
}

// CopyInto copies g into dst (same shape).
func (g *Grid) CopyInto(dst *Grid) {
	if dst == nil || len(dst.Data) != len(g.Data) {
		return
	}
	copy(dst.Data, g.Data)
}
