package quantum

// ExpectationH returns ⟨H⟩ under the product-state mean-field approximation:
//   Σ_s ⟨ψ_s|H_local|ψ_s⟩ + Σ_{⟨s,t⟩} J Σ_k ⟨X_{s,k}⟩⟨X_{t,k}⟩
// where the sum over edges counts each undirected bond once (x,y,z directions, all edges).
func (S *Simulator) ExpectationH() float64 {
	var eloc float64
	for s := 0; s < S.N; s++ {
		base := s * S.Dim
		eloc += siteExpectZ(S.ReA, S.ImA, base, S.Dim, S.Hz, S.NQ)
		for k := 0; k < S.NQ; k++ {
			eloc += S.Hx[k] * siteExpectX(S.ReA, S.ImA, base, S.Dim, k)
		}
	}
	var ebond float64
	lx, ly, lz := S.Lx, S.Ly, S.Lz
	for iz := 0; iz < lz; iz++ {
		for iy := 0; iy < ly; iy++ {
			for ix := 0; ix < lx; ix++ {
				i := S.IdxNode(ix, iy, iz)
				j := S.IdxNode(ix+1, iy, iz)
				for k := 0; k < S.NQ; k++ {
					ebond += S.JBond * S.SiteExpectXk(i, k) * S.SiteExpectXk(j, k)
				}
			}
		}
	}
	for iz := 0; iz < lz; iz++ {
		for ix := 0; ix < lx; ix++ {
			for iy := 0; iy < ly; iy++ {
				i := S.IdxNode(ix, iy, iz)
				j := S.IdxNode(ix, iy+1, iz)
				for k := 0; k < S.NQ; k++ {
					ebond += S.JBond * S.SiteExpectXk(i, k) * S.SiteExpectXk(j, k)
				}
			}
		}
	}
	for iy := 0; iy < ly; iy++ {
		for ix := 0; ix < lx; ix++ {
			for iz := 0; iz < lz; iz++ {
				i := S.IdxNode(ix, iy, iz)
				j := S.IdxNode(ix, iy, iz+1)
				for k := 0; k < S.NQ; k++ {
					ebond += S.JBond * S.SiteExpectXk(i, k) * S.SiteExpectXk(j, k)
				}
			}
		}
	}
	return eloc + ebond
}
