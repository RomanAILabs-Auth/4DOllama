package lattice4

// WaveState holds phi^n and phi^{n-1} plus a scratch buffer (no per-step heap allocs).
type WaveState struct {
	Phi     *Grid
	PhiPrev *Grid
	Scratch *Grid
	C       float64
	Dt      float64
}

// NewWaveState allocates three grids of identical shape.
func NewWaveState(nx, ny, nz, nw int, c, dt float64) *WaveState {
	return &WaveState{
		Phi:     NewGrid(nx, ny, nz, nw),
		PhiPrev: NewGrid(nx, ny, nz, nw),
		Scratch: NewGrid(nx, ny, nz, nw),
		C:       c,
		Dt:      dt,
	}
}

// InitQuiet sets phi^0 and phi^{-1} for quiet start (both same).
func (s *WaveState) InitQuiet() {
	if s.Phi == nil || s.PhiPrev == nil {
		return
	}
	copy(s.PhiPrev.Data, s.Phi.Data)
}

// StepLeapfrog advances one step: phi+ = 2*phi - phi- + (c*dt)^2 * Lap(phi) + source.
func (s *WaveState) StepLeapfrog(source *Grid) {
	if s.Phi == nil || s.PhiPrev == nil || s.Scratch == nil {
		return
	}
	nx, ny, nz, nw := s.Phi.Nx, s.Phi.Ny, s.Phi.Nz, s.Phi.Nw
	coef := (s.C * s.Dt) * (s.C * s.Dt)
	for iw := 0; iw < nw; iw++ {
		for iz := 0; iz < nz; iz++ {
			for iy := 0; iy < ny; iy++ {
				for ix := 0; ix < nx; ix++ {
					p := s.Phi.At(ix, iy, iz, iw)
					pm := s.PhiPrev.At(ix, iy, iz, iw)
					lap := s.Phi.Laplacian4(ix, iy, iz, iw)
					src := 0.0
					if source != nil {
						src = source.At(ix, iy, iz, iw)
					}
					// light numerical damping to keep explicit scheme bounded on coarse 4-torus demos
					damp := 0.999
					s.Scratch.Set(ix, iy, iz, iw, damp*(2*p-pm+coef*lap)+src)
				}
			}
		}
	}
	// rotate: prev <- cur, cur <- scratch, scratch <- prev
	s.PhiPrev, s.Phi, s.Scratch = s.Phi, s.Scratch, s.PhiPrev
}
