package lattice4

// StepHeat performs one explicit heat / diffusion step (stable for small D*dt on periodic grid):
// phi^{n+1} = phi^n + D*dt*Lap(phi^n) + source.
func StepHeat(phi, next, source *Grid, D, dt float64) {
	if phi == nil || next == nil {
		return
	}
	nx, ny, nz, nw := phi.Nx, phi.Ny, phi.Nz, phi.Nw
	alpha := D * dt
	for iw := 0; iw < nw; iw++ {
		for iz := 0; iz < nz; iz++ {
			for iy := 0; iy < ny; iy++ {
				for ix := 0; ix < nx; ix++ {
					lap := phi.Laplacian4(ix, iy, iz, iw)
					v := phi.At(ix, iy, iz, iw) + alpha*lap
					if source != nil {
						v += source.At(ix, iy, iz, iw)
					}
					next.Set(ix, iy, iz, iw, v)
				}
			}
		}
	}
}
