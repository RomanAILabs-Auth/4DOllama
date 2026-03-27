package quantum

import "math"

// maxDim is the maximum local Hilbert dimension (sites use d ≤ maxDim).
const maxDim = 8

// Backend names (CLI / scripts).
const (
	BackendMeanField = "meanfield"
	BackendTN        = "tn"
	BackendCPU       = "cpu" // alias for meanfield (legacy)
)

// MaxChiCap is the maximum χ stored per bond (memory ∝ 3·N·MaxChiCap·d).
const MaxChiCap = 32

// BondState stores truncated Schmidt data for one canonical directed edge (+x/+y/+z).
// SingularValues[k] are Schmidt weights w_k ≥ 0 with Σ w_k^2 = 1 after renormalization.
// Left*/Right* hold U and V as d×Chi row-major U[a,k] at index a*d+k.
type BondState struct {
	Chi              int
	SingularValues   [MaxChiCap]float64
	LeftRe, LeftIm   [maxDim * MaxChiCap]float64
	RightRe, RightIm [maxDim * MaxChiCap]float64
}

func (b *BondState) clear() { b.Chi = 0 }

// EdgeIndexX is the bond slot for the +x edge from (ix,iy,iz).
func (S *Simulator) EdgeIndexX(ix, iy, iz int) int { return 0*S.N + S.IdxNode(ix, iy, iz) }
func (S *Simulator) EdgeIndexY(ix, iy, iz int) int { return 1*S.N + S.IdxNode(ix, iy, iz) }
func (S *Simulator) EdgeIndexZ(ix, iy, iz int) int { return 2*S.N + S.IdxNode(ix, iy, iz) }

func (S *Simulator) isTNMultBond() bool { return S.Backend == BackendTN && S.Chi > 1 }

// initTNStorage allocates rho double-buffer and 3N bond slots when Backend==tn.
func (S *Simulator) initTNStorage() {
	if S.Backend != BackendTN {
		return
	}
	dd := S.Dim * S.Dim
	n := S.N
	S.RhoA = make([]float64, n*dd*2)
	S.RhoB = make([]float64, n*dd*2)
	S.Bonds = make([]BondState, 3*n)
}

// InitRhoComputational sets each site's reduced state to |0…0⟩⟨0…0|.
func (S *Simulator) InitRhoComputational() {
	if S.RhoA == nil {
		return
	}
	d := S.Dim
	dd := d * d
	n := S.N
	clear(S.RhoA)
	clear(S.RhoB)
	for s := 0; s < n; s++ {
		o := s * dd * 2
		S.RhoA[o] = 1
		S.RhoB[o] = 1
	}
}

func rhoOffset(site, d int) int {
	dd := d * d
	return site * dd * 2
}

func rhoPtrFlat(flat []float64, site, d int) ([]float64, []float64) {
	o := rhoOffset(site, d)
	dd := d * d
	return flat[o : o+dd], flat[o+dd : o+2*dd]
}

// gramMHM forms G = M^H M (Hermitian PSD); M row-major M[a,b] = m[a*d+b].
func gramMHM(mRe, mIm []float64, d int, gRe, gIm []float64) {
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			var sr, si float64
			for k := 0; k < d; k++ {
				mkir, mkii := mRe[k*d+i], mIm[k*d+i]
				mkjr, mkji := mRe[k*d+j], mIm[k*d+j]
				sr += mkir*mkjr + mkii*mkji
				si += mkir*mkji - mkii*mkjr
			}
			gRe[i*d+j] = sr
			gIm[i*d+j] = si
		}
	}
}

func frobNorm2Sq(re, im []float64, d int) float64 {
	var s float64
	for i := 0; i < d*d; i++ {
		s += re[i]*re[i] + im[i]*im[i]
	}
	return s
}

func scaleEta(re, im []float64, d int) {
	n2 := frobNorm2Sq(re, im, d)
	if n2 <= 1e-30 {
		return
	}
	inv := 1.0 / math.Sqrt(n2)
	for i := 0; i < d*d; i++ {
		re[i] *= inv
		im[i] *= inv
	}
}

// rhoHermitianFromSchmidt builds ρ = Σ_k w_k^2 |u_k⟩⟨u_k| (columns u_k orthonormal).
func rhoHermitianFromSchmidt(d, chi int, w []float64, uRe, uIm []float64, rhoRe, rhoIm []float64) {
	dd := d * d
	clear(rhoRe[:dd])
	clear(rhoIm[:dd])
	for k := 0; k < chi; k++ {
		p := w[k] * w[k]
		if p <= 1e-30 {
			continue
		}
		for a := 0; a < d; a++ {
			uaR, uaI := uRe[a*d+k], uIm[a*d+k]
			for b := 0; b < d; b++ {
				ubR, ubI := uRe[b*d+k], uIm[b*d+k]
				rhoRe[a*d+b] += p * (uaR*ubR + uaI*ubI)
				rhoIm[a*d+b] += p * (uaI*ubR - uaR*ubI)
			}
		}
	}
}

// topEigenpairsDeflated extracts the largest eigenpairs of Hermitian G by power iteration + deflation.
// Columns k of vRe/vIm are eigenvectors; eval[k] are eigenvalues (descending if spectrum allows).
func topEigenpairsDeflated(d, chi int, gRe, gIm []float64,
	workRe, workIm []float64,
	vRe, vIm []float64, eval []float64,
	tmpVre, tmpVim []float64) int {

	var wRe, wIm [maxDim]float64
	copy(workRe, gRe[:d*d])
	copy(workIm, gIm[:d*d])
	nFound := 0
	maxChi := chi
	if maxChi > d {
		maxChi = d
	}
	for k := 0; k < maxChi; k++ {
		for i := 0; i < d; i++ {
			tmpVre[i] = math.Sin(float64(i+k+1)*0.713) * 0.1
			tmpVim[i] = math.Cos(float64(i+k+2)*0.519) * 0.1
		}
		normV(tmpVre, tmpVim, d)
		for iter := 0; iter < 48; iter++ {
			hermMatVec(workRe, workIm, d, tmpVre, tmpVim, wRe[:], wIm[:])
			copy(tmpVre, wRe[:d])
			copy(tmpVim, wIm[:d])
			normV(tmpVre, tmpVim, d)
		}
		lam := rayleighQuotient(workRe, workIm, d, tmpVre, tmpVim)
		if lam < 1e-18 {
			break
		}
		eval[k] = lam
		for i := 0; i < d; i++ {
			vRe[i+k*d] = tmpVre[i]
			vIm[i+k*d] = tmpVim[i]
		}
		nFound++
		deflateHermitian(workRe, workIm, d, lam, tmpVre, tmpVim)
	}
	return nFound
}

func normV(vr, vi []float64, d int) {
	var s float64
	for i := 0; i < d; i++ {
		s += vr[i]*vr[i] + vi[i]*vi[i]
	}
	if s <= 1e-30 {
		return
	}
	inv := 1.0 / math.Sqrt(s)
	for i := 0; i < d; i++ {
		vr[i] *= inv
		vi[i] *= inv
	}
}

func hermMatVec(hRe, hIm []float64, d int, vRe, vIm []float64, outRe, outIm []float64) {
	for i := 0; i < d; i++ {
		var sr, si float64
		for j := 0; j < d; j++ {
			hr, hi := hRe[i*d+j], hIm[i*d+j]
			vr, vi := vRe[j], vIm[j]
			sr += hr*vr - hi*vi
			si += hr*vi + hi*vr
		}
		outRe[i] = sr
		outIm[i] = si
	}
}

func rayleighQuotient(hRe, hIm []float64, d int, vRe, vIm []float64) float64 {
	var hvRe, hvIm [maxDim]float64
	hermMatVec(hRe, hIm, d, vRe, vIm, hvRe[:], hvIm[:])
	var r1r, r1i float64
	for i := 0; i < d; i++ {
		r1r += vRe[i]*hvRe[i] + vIm[i]*hvIm[i]
		r1i += vRe[i]*hvIm[i] - vIm[i]*hvRe[i]
	}
	return r1r // imaginary ~0 for Hermitian
}

func deflateHermitian(hRe, hIm []float64, d int, lam float64, vRe, vIm []float64) {
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			viR, viI := vRe[i], vIm[i]
			vjR, vjI := vRe[j], vIm[j]
			// (v v^H)_ij = v_i conj(v_j)
			rr := viR*vjR + viI*vjI
			ri := viI*vjR - viR*vjI
			hRe[i*d+j] -= lam * rr
			hIm[i*d+j] -= lam * ri
		}
	}
}

// svdTruncatedFromM: normalized M (Frobenius 1) → top χ Schmidt weights and U,V columns.
// sigmaOut[k] = Schmidt weight (renormalized truncated). Returns effective chi.
func svdTruncatedFromM(mRe, mIm []float64, d, chiCap int,
	gRe, gIm []float64, workRe, workIm []float64,
	vRe, vIm []float64, eval []float64,
	tmpVre, tmpVim []float64,
	uRe, uIm []float64, sigmaOut []float64) int {

	gramMHM(mRe, mIm, d, gRe, gIm)
	nEig := topEigenpairsDeflated(d, chiCap, gRe, gIm, workRe, workIm, vRe, vIm, eval, tmpVre, tmpVim)
	if nEig < 1 {
		return 0
	}
	chiEff := nEig
	if chiEff > chiCap {
		chiEff = chiCap
	}
	// σ_k = sqrt(λ_k), U_k = (1/σ_k) M v_k
	var sumW2 float64
	for k := 0; k < chiEff; k++ {
		lam := eval[k]
		if lam < 0 {
			lam = 0
		}
		sg := math.Sqrt(lam)
		sigmaOut[k] = sg
		sumW2 += sg * sg
		for a := 0; a < d; a++ {
			var sr, si float64
			for b := 0; b < d; b++ {
				mr, mi := mRe[a*d+b], mIm[a*d+b]
				vr, vi := vRe[b*d+k], vIm[b*d+k]
				sr += mr*vr - mi*vi
				si += mr*vi + mi*vr
			}
			if sg > 1e-14 {
				uRe[a+k*d] = sr / sg
				uIm[a+k*d] = si / sg
			} else {
				uRe[a+k*d], uIm[a+k*d] = 0, 0
			}
		}
	}
	// Renormalize truncated weights so Σ w_k^2 = 1
	if sumW2 <= 1e-30 {
		return 0
	}
	inv := 1.0 / math.Sqrt(sumW2)
	for k := 0; k < chiEff; k++ {
		sigmaOut[k] *= inv
	}
	return chiEff
}
