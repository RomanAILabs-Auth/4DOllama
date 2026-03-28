package quantum

import "math/rand/v2"

// MeasureSite samples a computational outcome k. For BackendTN with ρ allocated,
// p(k)=ρ_kk (computational-basis diagonal, Tr(ρ)=1). Otherwise p(k)=|α_k|²/‖ψ‖².
// Collapse updates ρ (TN) or ψ (mean-field); TN collapse clears incident BondState entries.
func (S *Simulator) MeasureSite(site int, rng *rand.Rand, collapse bool) int {
	if S.Backend == BackendTN && S.RhoA != nil {
		return S.measureSiteRho(site, rng, collapse)
	}
	d := S.Dim
	base := site * d
	var sum float64
	for k := 0; k < d; k++ {
		r, i := S.ReA[base+k], S.ImA[base+k]
		sum += r*r + i*i
	}
	if sum <= 1e-30 {
		return 0
	}
	u := rng.Float64() * sum
	var acc float64
	chosen := d - 1
	for k := 0; k < d; k++ {
		r, i := S.ReA[base+k], S.ImA[base+k]
		p := r*r + i*i
		acc += p
		if u < acc {
			chosen = k
			break
		}
	}
	if collapse {
		for k := 0; k < d; k++ {
			S.ReA[base+k] = 0
			S.ImA[base+k] = 0
		}
		S.ReA[base+chosen] = 1
		S.ImA[base+chosen] = 0
	}
	return chosen
}

func (S *Simulator) measureSiteRho(site int, rng *rand.Rand, collapse bool) int {
	d := S.Dim
	rre, rim := rhoPtrFlat(S.RhoA, site, d)
	var sum float64
	for k := 0; k < d; k++ {
		sum += rre[k*d+k]
	}
	if sum <= 1e-30 {
		return 0
	}
	u := rng.Float64() * sum
	var acc float64
	chosen := d - 1
	for k := 0; k < d; k++ {
		p := rre[k*d+k]
		if p < 0 {
			p = 0
		}
		acc += p
		if u < acc {
			chosen = k
			break
		}
	}
	if collapse {
		dd := d * d
		for i := 0; i < dd; i++ {
			rre[i] = 0
			rim[i] = 0
		}
		rre[chosen*d+chosen] = 1
		if S.RhoB != nil {
			br, bi := rhoPtrFlat(S.RhoB, site, d)
			for i := 0; i < dd; i++ {
				br[i], bi[i] = 0, 0
			}
			br[chosen*d+chosen] = 1
		}
		S.invalidateBondsAtSite(site)
		base := site * d
		for k := 0; k < d; k++ {
			S.ReA[base+k] = 0
			S.ImA[base+k] = 0
		}
		S.ReA[base+chosen] = 1
		S.ImA[base+chosen] = 0
	}
	return chosen
}

func (S *Simulator) invalidateBondsAtSite(site int) {
	if S.Bonds == nil {
		return
	}
	ix, iy, iz := S.FlatToGrid(site)
	lx, ly, lz := S.Lx, S.Ly, S.Lz
	xm, ym, zm := ix-1, iy-1, iz-1
	if xm < 0 {
		xm += lx
	}
	if ym < 0 {
		ym += ly
	}
	if zm < 0 {
		zm += lz
	}
	S.Bonds[S.EdgeIndexX(ix, iy, iz)].clear()
	S.Bonds[S.EdgeIndexX(xm, iy, iz)].clear()
	S.Bonds[S.EdgeIndexY(ix, iy, iz)].clear()
	S.Bonds[S.EdgeIndexY(ix, ym, iz)].clear()
	S.Bonds[S.EdgeIndexZ(ix, iy, iz)].clear()
	S.Bonds[S.EdgeIndexZ(ix, iy, zm)].clear()
}

// BatchMeasure runs MeasureSite on sites 0, stride, 2*stride, … up to count samples.
func (S *Simulator) BatchMeasure(rng *rand.Rand, count, stride int, collapse bool) []int {
	if stride < 1 {
		stride = 1
	}
	out := make([]int, 0, count)
	for n := 0; n < count; n++ {
		s := n * stride
		if s >= S.N {
			break
		}
		out = append(out, S.MeasureSite(s, rng, collapse))
	}
	return out
}

// SiteNorm returns ‖ψ_site‖₂², or Tr(ρ_site) when TN uses reduced density matrices.
func (S *Simulator) SiteNorm(site int) float64 {
	d := S.Dim
	if S.Backend == BackendTN && S.RhoA != nil {
		rre, _ := rhoPtrFlat(S.RhoA, site, d)
		var t float64
		for k := 0; k < d; k++ {
			t += rre[k*d+k]
		}
		return t
	}
	base := site * d
	var s float64
	for k := 0; k < d; k++ {
		r, i := S.ReA[base+k], S.ImA[base+k]
		s += r*r + i*i
	}
	return s
}

// ProbK returns marginal probability of outcome k (ρ_kk for TN, else |α_k|²/‖ψ‖²).
func (S *Simulator) ProbK(site, k int) float64 {
	d := S.Dim
	if k < 0 || k >= d {
		return 0
	}
	if S.Backend == BackendTN && S.RhoA != nil {
		rre, _ := rhoPtrFlat(S.RhoA, site, d)
		return rre[k*d+k]
	}
	base := site * d
	var sum float64
	for j := 0; j < d; j++ {
		r, i := S.ReA[base+j], S.ImA[base+j]
		sum += r*r + i*i
	}
	if sum <= 1e-30 {
		return 0
	}
	r, i := S.ReA[base+k], S.ImA[base+k]
	return (r*r + i*i) / sum
}

// NewRNG returns *rand.Rand from a single 64-bit seed (PCG).
func NewRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewPCG(uint64(seed), 0x123456789abcdef0))
}

// --- helpers for energy ---
func siteExpectX(re, im []float64, base, d, k int) float64 {
	stride := 1 << k
	var ex float64
	for b0 := 0; b0 < d; b0 += 2 * stride {
		for off := 0; off < stride; off++ {
			i0 := base + b0 + off
			i1 := base + b0 + off + stride
			ex += 2 * (re[i0]*re[i1] + im[i0]*im[i1])
		}
	}
	return ex
}

func siteExpectZ(re, im []float64, base int, d int, hz []float64, nq int) float64 {
	var ez float64
	for idx := 0; idx < d; idx++ {
		p := re[base+idx]*re[base+idx] + im[base+idx]*im[base+idx]
		ez += p * sumZexpect(idx, hz, nq)
	}
	return ez
}

func sumZexpect(idx int, hz []float64, nq int) float64 {
	var ph float64
	for k := 0; k < nq; k++ {
		bit := (idx >> k) & 1
		sign := 1.0
		if bit != 0 {
			sign = -1.0
		}
		ph += hz[k] * sign
	}
	return ph
}

// SiteExpectZ returns ⟨ψ_site| Σ_k hz_k Z_k |ψ_site⟩ (not normalized by site norm).
func (S *Simulator) SiteExpectZ(site int) float64 {
	return siteExpectZ(S.ReA, S.ImA, site*S.Dim, S.Dim, S.Hz, S.NQ)
}

// SiteExpectXk returns ⟨X_k⟩ at the site.
func (S *Simulator) SiteExpectXk(site, k int) float64 {
	return siteExpectX(S.ReA, S.ImA, site*S.Dim, S.Dim, k)
}
