package quantum

import (
	"math"
	"runtime"
	"sync"
)

// Step runs one Trotter timestep. Mean-field path is unchanged from legacy behavior.
// TN path with χ>1 applies truncated Schmidt bonds and maintains ρ on each site.
func (S *Simulator) Step() {
	if S.isTNMultBond() {
		S.stepTNMulti()
		return
	}
	S.stepMeanField()
	if S.Backend == BackendTN && S.RhoA != nil {
		S.SyncAllRhoFromPsi(S.ReA, S.ImA)
		copy(S.RhoB, S.RhoA)
	}
}

func (S *Simulator) stepMeanField() {
	w := S.Workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	if w < 1 {
		w = 1
	}
	readRe, readIm := S.ReA, S.ImA
	writeRe, writeIm := S.ReB, S.ImB
	S.applyLocalStrangParallel(readRe, readIm, writeRe, writeIm, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsX(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsX(readRe, readIm, writeRe, writeIm, false, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsY(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsY(readRe, readIm, writeRe, writeIm, false, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsZ(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.evolveBondsZ(readRe, readIm, writeRe, writeIm, false, w)
	S.ReA, S.ImA, S.ReB, S.ImB = readRe, readIm, writeRe, writeIm
}

func (S *Simulator) stepTNMulti() {
	w := S.Workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	if w < 1 {
		w = 1
	}
	S.rhoStepRead, S.rhoStepWrite = S.RhoA, S.RhoB
	readRe, readIm := S.ReA, S.ImA
	writeRe, writeIm := S.ReB, S.ImB
	S.applyLocalStrangParallel(readRe, readIm, writeRe, writeIm, w)
	S.applyLocalStrangRhoParallel(S.rhoStepRead, S.rhoStepWrite, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead

	S.evolveBondsX(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead
	S.evolveBondsX(readRe, readIm, writeRe, writeIm, false, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead
	S.evolveBondsY(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead
	S.evolveBondsY(readRe, readIm, writeRe, writeIm, false, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead
	S.evolveBondsZ(readRe, readIm, writeRe, writeIm, true, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead
	S.evolveBondsZ(readRe, readIm, writeRe, writeIm, false, w)
	readRe, readIm, writeRe, writeIm = writeRe, writeIm, readRe, readIm
	S.rhoStepRead, S.rhoStepWrite = S.rhoStepWrite, S.rhoStepRead

	S.ReA, S.ImA, S.ReB, S.ImB = readRe, readIm, writeRe, writeIm
	S.RhoA, S.RhoB = S.rhoStepRead, S.rhoStepWrite
}

func (S *Simulator) copyRho(src, dst []float64) {
	n := S.N * S.Dim * S.Dim * 2
	copy(dst[:n], src[:n])
}

func (S *Simulator) applyLocalStrangParallel(srcRe, srcIm, dstRe, dstIm []float64, workers int) {
	d := S.Dim
	n := S.N
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for t := 0; t < workers; t++ {
		i0 := t * chunk
		i1 := i0 + chunk
		if i1 > n {
			i1 = n
		}
		if i0 >= i1 {
			break
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			for s := a; s < b; s++ {
				base := s * d
				copy(dstRe[base:base+d], srcRe[base:base+d])
				copy(dstIm[base:base+d], srcIm[base:base+d])
				applyZPhaseSite(dstRe, dstIm, base, d, S.Hz, S.NQ, S.Dt*0.5)
				for k := 0; k < S.NQ; k++ {
					applyPauliXLayer(dstRe, dstIm, base, d, k, -S.Dt*S.Hx[k])
				}
				applyZPhaseSite(dstRe, dstIm, base, d, S.Hz, S.NQ, S.Dt*0.5)
			}
		}(i0, i1)
	}
	wg.Wait()
}

func applyZPhaseSite(re, im []float64, base, d int, hz []float64, nq int, dt float64) {
	for a := 0; a < d; a++ {
		var phase float64
		for k := 0; k < nq; k++ {
			bit := (a >> k) & 1
			sign := 1.0
			if bit != 0 {
				sign = -1.0
			}
			phase += hz[k] * sign
		}
		phi := -dt * phase
		c, s := math.Cos(phi), math.Sin(phi)
		r, imc := re[base+a], im[base+a]
		re[base+a] = r*c - imc*s
		im[base+a] = r*s + imc*c
	}
}

func applyPauliXLayer(re, im []float64, base, d, k int, angle float64) {
	stride := 1 << k
	cosA, sinA := math.Cos(angle), math.Sin(angle)
	for b0 := 0; b0 < d; b0 += 2 * stride {
		for off := 0; off < stride; off++ {
			i0 := base + b0 + off
			i1 := base + b0 + off + stride
			r0, i0c := re[i0], im[i0]
			r1, i1c := re[i1], im[i1]
			re[i0] = cosA*r0 + sinA*i1c
			im[i0] = cosA*i0c - sinA*r1
			re[i1] = sinA*i0c + cosA*r1
			im[i1] = -sinA*r0 + cosA*i1c
		}
	}
}

// --- Bond evolution (XX layers + Schmidt rank-1 back to product form) ---

func (S *Simulator) evolveBondsX(srcRe, srcIm, dstRe, dstIm []float64, ixEven bool, workers int) {
	S.copyState(srcRe, srcIm, dstRe, dstIm)
	if S.isTNMultBond() {
		S.copyRho(S.rhoStepRead, S.rhoStepWrite)
	}
	lx, ly, lz := S.Lx, S.Ly, S.Lz
	var wg sync.WaitGroup
	chunk := (lz + workers - 1) / workers
	for t := 0; t < workers; t++ {
		z0 := t * chunk
		z1 := z0 + chunk
		if z1 > lz {
			z1 = lz
		}
		if z0 >= z1 {
			break
		}
		wg.Add(1)
		go func(za, zb int) {
			defer wg.Done()
			d := S.Dim
			d2 := d * d
			var etaRe, etaIm [maxBondTensor]float64
			var uR, uI, vR, vI [maxDim]float64
			var t1R, t1I, t2R, t2I [maxDim]float64
			for iz := za; iz < zb; iz++ {
				for iy := 0; iy < ly; iy++ {
					for ix := 0; ix < lx; ix++ {
						if (ix%2 == 0) != ixEven {
							continue
						}
						i := S.IdxNode(ix, iy, iz)
						j := S.IdxNode(ix+1, iy, iz)
						eIdx := S.EdgeIndexX(ix, iy, iz)
						if S.isTNMultBond() {
							S.bondUpdatePairTN(srcRe, srcIm, dstRe, dstIm, S.rhoStepWrite, i, j, eIdx, etaRe[:d2], etaIm[:d2])
						} else {
							S.bondUpdatePair(srcRe, srcIm, dstRe, dstIm, i, j,
								etaRe[:d2], etaIm[:d2],
								uR[:d], uI[:d], vR[:d], vI[:d],
								t1R[:d], t1I[:d], t2R[:d], t2I[:d])
						}
					}
				}
			}
		}(z0, z1)
	}
	wg.Wait()
}

const maxBondTensor = 64

func (S *Simulator) evolveBondsY(srcRe, srcIm, dstRe, dstIm []float64, iyEven bool, workers int) {
	S.copyState(srcRe, srcIm, dstRe, dstIm)
	if S.isTNMultBond() {
		S.copyRho(S.rhoStepRead, S.rhoStepWrite)
	}
	lx, ly, lz := S.Lx, S.Ly, S.Lz
	var wg sync.WaitGroup
	chunk := (lz + workers - 1) / workers
	for t := 0; t < workers; t++ {
		z0 := t * chunk
		z1 := z0 + chunk
		if z1 > lz {
			z1 = lz
		}
		if z0 >= z1 {
			break
		}
		wg.Add(1)
		go func(za, zb int) {
			defer wg.Done()
			d := S.Dim
			d2 := d * d
			var etaRe, etaIm [maxBondTensor]float64
			var uR, uI, vR, vI [maxDim]float64
			var t1R, t1I, t2R, t2I [maxDim]float64
			for iz := za; iz < zb; iz++ {
				for ix := 0; ix < lx; ix++ {
					for iy := 0; iy < ly; iy++ {
						if (iy%2 == 0) != iyEven {
							continue
						}
						i := S.IdxNode(ix, iy, iz)
						j := S.IdxNode(ix, iy+1, iz)
						eIdx := S.EdgeIndexY(ix, iy, iz)
						if S.isTNMultBond() {
							S.bondUpdatePairTN(srcRe, srcIm, dstRe, dstIm, S.rhoStepWrite, i, j, eIdx, etaRe[:d2], etaIm[:d2])
						} else {
							S.bondUpdatePair(srcRe, srcIm, dstRe, dstIm, i, j,
								etaRe[:d2], etaIm[:d2],
								uR[:d], uI[:d], vR[:d], vI[:d],
								t1R[:d], t1I[:d], t2R[:d], t2I[:d])
						}
					}
				}
			}
		}(z0, z1)
	}
	wg.Wait()
}

func (S *Simulator) evolveBondsZ(srcRe, srcIm, dstRe, dstIm []float64, izEven bool, workers int) {
	S.copyState(srcRe, srcIm, dstRe, dstIm)
	if S.isTNMultBond() {
		S.copyRho(S.rhoStepRead, S.rhoStepWrite)
	}
	lx, ly, lz := S.Lx, S.Ly, S.Lz
	var wg sync.WaitGroup
	chunk := (ly + workers - 1) / workers
	for t := 0; t < workers; t++ {
		y0 := t * chunk
		y1 := y0 + chunk
		if y1 > ly {
			y1 = ly
		}
		if y0 >= y1 {
			break
		}
		wg.Add(1)
		go func(ya, yb int) {
			defer wg.Done()
			d := S.Dim
			d2 := d * d
			var etaRe, etaIm [maxBondTensor]float64
			var uR, uI, vR, vI [maxDim]float64
			var t1R, t1I, t2R, t2I [maxDim]float64
			for iy := ya; iy < yb; iy++ {
				for ix := 0; ix < lx; ix++ {
					for iz := 0; iz < lz; iz++ {
						if (iz%2 == 0) != izEven {
							continue
						}
						i := S.IdxNode(ix, iy, iz)
						j := S.IdxNode(ix, iy, iz+1)
						eIdx := S.EdgeIndexZ(ix, iy, iz)
						if S.isTNMultBond() {
							S.bondUpdatePairTN(srcRe, srcIm, dstRe, dstIm, S.rhoStepWrite, i, j, eIdx, etaRe[:d2], etaIm[:d2])
						} else {
							S.bondUpdatePair(srcRe, srcIm, dstRe, dstIm, i, j,
								etaRe[:d2], etaIm[:d2],
								uR[:d], uI[:d], vR[:d], vI[:d],
								t1R[:d], t1I[:d], t2R[:d], t2I[:d])
						}
					}
				}
			}
		}(y0, y1)
	}
	wg.Wait()
}

func (S *Simulator) copyState(srcRe, srcIm, dstRe, dstIm []float64) {
	n := S.N * S.Dim
	copy(dstRe[:n], srcRe[:n])
	copy(dstIm[:n], srcIm[:n])
}

func (S *Simulator) bondUpdatePair(srcRe, srcIm, dstRe, dstIm []float64, si, sj int,
	etaRe, etaIm []float64,
	uR, uI, vR, vI []float64,
	t1R, t1I, t2R, t2I []float64) {

	d := S.Dim
	bi := si * d
	bj := sj * d
	sPair0 := siteL2Sq(srcRe, srcIm, bi, d) + siteL2Sq(srcRe, srcIm, bj, d)

	// η_{ab} = ψ_i(a) ψ_j(b)
	for a := 0; a < d; a++ {
		for b := 0; b < d; b++ {
			idx := a*d + b
			ar, ai := srcRe[bi+a], srcIm[bi+a]
			br, biM := srcRe[bj+b], srcIm[bj+b]
			etaRe[idx] = ar*br - ai*biM
			etaIm[idx] = ar*biM + ai*br
		}
	}

	theta := -S.Dt * S.JBond
	for k := 0; k < S.NQ; k++ {
		applyXXOnBondTensor(etaRe, etaIm, d, k, theta)
	}

	sigmaSq := rank1Schmidt(etaRe, etaIm, d, uR, uI, vR, vI, t1R, t1I, t2R, t2I, 24)
	singular := math.Sqrt(sigmaSq)
	scale := math.Sqrt(singular)
	if sigmaSq < 1e-30 || math.IsNaN(sigmaSq) {
		scale = 0
	}
	for a := 0; a < d; a++ {
		dstRe[bi+a] = scale * uR[a]
		dstIm[bi+a] = scale * uI[a]
		dstRe[bj+a] = scale * vR[a]
		dstIm[bj+a] = scale * vI[a]
	}
	// Preserve Σ_k ‖ψ_k‖² over the two sites (global concatenated norm) after rank-1 truncation.
	sPair1 := siteL2Sq(dstRe, dstIm, bi, d) + siteL2Sq(dstRe, dstIm, bj, d)
	if sPair1 > 1e-30 && sPair0 > 0 {
		m := math.Sqrt(sPair0 / sPair1)
		for a := 0; a < d; a++ {
			dstRe[bi+a] *= m
			dstIm[bi+a] *= m
			dstRe[bj+a] *= m
			dstIm[bj+a] *= m
		}
	}
}

func (S *Simulator) bondUpdatePairTN(srcRe, srcIm, dstRe, dstIm []float64, dstRho []float64, si, sj, edgeIdx int, etaRe, etaIm []float64) {
	d := S.Dim
	bi := si * d
	bj := sj * d
	for a := 0; a < d; a++ {
		for b := 0; b < d; b++ {
			idx := a*d + b
			ar, ai := srcRe[bi+a], srcIm[bi+a]
			br, bim := srcRe[bj+b], srcIm[bj+b]
			etaRe[idx] = ar*br - ai*bim
			etaIm[idx] = ar*bim + ai*br
		}
	}
	theta := -S.Dt * S.JBond
	for k := 0; k < S.NQ; k++ {
		applyXXOnBondTensor(etaRe, etaIm, d, k, theta)
	}
	scaleEta(etaRe, etaIm, d)

	chiCap := S.Chi
	if chiCap > d {
		chiCap = d
	}
	var gRe, gIm, workRe, workIm [maxBondTensor]float64
	var vRe, vIm [maxDim * maxDim]float64
	var uRe, uIm [maxDim * maxDim]float64
	var eval [maxDim]float64
	var tmpVre, tmpVim [maxDim]float64
	var sigma [MaxChiCap]float64

	chiEff := svdTruncatedFromM(etaRe, etaIm, d, chiCap,
		gRe[:d*d], gIm[:d*d], workRe[:d*d], workIm[:d*d],
		vRe[:d*d], vIm[:d*d], eval[:d], tmpVre[:d], tmpVim[:d],
		uRe[:d*d], uIm[:d*d], sigma[:])

	b := &S.Bonds[edgeIdx]
	b.clear()
	if chiEff < 1 {
		return
	}
	b.Chi = chiEff
	for k := 0; k < chiEff; k++ {
		b.SingularValues[k] = sigma[k]
		for a := 0; a < d; a++ {
			b.LeftRe[a*d+k] = uRe[a*d+k]
			b.LeftIm[a*d+k] = uIm[a*d+k]
			b.RightRe[a*d+k] = vRe[a*d+k]
			b.RightIm[a*d+k] = vIm[a*d+k]
		}
	}

	lr, li := rhoPtrFlat(dstRho, si, d)
	rr, ri := rhoPtrFlat(dstRho, sj, d)
	rhoHermitianFromSchmidt(d, chiEff, sigma[:chiEff], uRe[:], uIm[:], lr, li)
	rhoHermitianFromSchmidt(d, chiEff, sigma[:chiEff], vRe[:], vIm[:], rr, ri)

	psiFromRhoIntoSite(dstRho, si, d, dstRe, dstIm)
	psiFromRhoIntoSite(dstRho, sj, d, dstRe, dstIm)
}

func siteL2Sq(re, im []float64, base, d int) float64 {
	var s float64
	for k := 0; k < d; k++ {
		r, i := re[base+k], im[base+k]
		s += r*r + i*i
	}
	return s
}

// ψ_i = √σ u,  ψ_j = √σ conj(v)  so ψ_i(a)ψ_j(b) ≈ σ u_a conj(v_b).
func rank1Schmidt(etaRe, etaIm []float64, d int,
	uR, uI, vR, vI []float64,
	t1R, t1I, t2R, t2I []float64, iters int) float64 {

	// Initialize v = uniform
	nrm := 1.0 / math.Sqrt(float64(d))
	for b := 0; b < d; b++ {
		vR[b] = nrm
		vI[b] = 0
	}
	for iter := 0; iter < iters; iter++ {
		// t1 = M v  (M_{ab} = eta_{ab}, row a col b)
		for a := 0; a < d; a++ {
			var sr, si float64
			for b := 0; b < d; b++ {
				idx := a*d + b
				er, ei := etaRe[idx], etaIm[idx]
				vr, vi := vR[b], vI[b]
				sr += er*vr - ei*vi
				si += er*vi + ei*vr
			}
			t1R[a], t1I[a] = sr, si
		}
		// t2 = M^H t1
		for b := 0; b < d; b++ {
			var sr, si float64
			for a := 0; a < d; a++ {
				idx := a*d + b
				er, ei := etaRe[idx], etaIm[idx]
				wr, wi := t1R[a], t1I[a]
				sr += er*wr + ei*wi
				si += ei*wr - er*wi
			}
			t2R[b], t2I[b] = sr, si
		}
		var nn float64
		for b := 0; b < d; b++ {
			nn += t2R[b]*t2R[b] + t2I[b]*t2I[b]
		}
		if nn <= 1e-300 {
			break
		}
		inv := 1.0 / math.Sqrt(nn)
		for b := 0; b < d; b++ {
			vR[b] = t2R[b] * inv
			vI[b] = t2I[b] * inv
		}
	}
	// u = M v / ||M v||
	for a := 0; a < d; a++ {
		var sr, si float64
		for b := 0; b < d; b++ {
			idx := a*d + b
			er, ei := etaRe[idx], etaIm[idx]
			vr, vi := vR[b], vI[b]
			sr += er*vr - ei*vi
			si += er*vi + ei*vr
		}
		t1R[a], t1I[a] = sr, si
	}
	var nMu float64
	for a := 0; a < d; a++ {
		nMu += t1R[a]*t1R[a] + t1I[a]*t1I[a]
	}
	if nMu <= 1e-300 {
		return 0
	}
	invMu := 1.0 / math.Sqrt(nMu)
	for a := 0; a < d; a++ {
		uR[a] = t1R[a] * invMu
		uI[a] = t1I[a] * invMu
	}
	sigma := nMu
	// v is right singular vector (already in vR,vI); ψ_j uses conj(v)
	for b := 0; b < d; b++ {
		vR[b], vI[b] = vR[b], -vI[b]
	}
	return sigma
}

func applyXXOnBondTensor(etaRe, etaIm []float64, d, k int, theta float64) {
	cosT, sinT := math.Cos(theta), math.Sin(theta)
	dk := 1 << k
	for a := 0; a < d; a++ {
		if (a>>k)&1 != 0 {
			continue
		}
		for b := 0; b < d; b++ {
			if (b>>k)&1 != 0 {
				continue
			}
			i00 := a*d + b
			i01 := a*d + (b + dk)
			i10 := (a + dk)*d + b
			i11 := (a + dk)*d + (b + dk)
			applyXX4(etaRe, etaIm, i00, i01, i10, i11, cosT, sinT)
		}
	}
}

func applyXX4(re, im []float64, i00, i01, i10, i11 int, cosT, sinT float64) {
	r0, i0 := re[i00], im[i00]
	r1, i1 := re[i01], im[i01]
	r2, i2 := re[i10], im[i10]
	r3, i3 := re[i11], im[i11]
	// Basis order (00,01,10,11) = (i00,i01,i10,i11). XX permutes 00↔11, 01↔10.
	// U = cos(θ) I - i sin(θ) XX  →  out = cos·ψ - i·sin·(XX·ψ)
	w0r, w0i := r3, i3
	w1r, w1i := r2, i2
	w2r, w2i := r1, i1
	w3r, w3i := r0, i0
	re[i00] = cosT*r0 + sinT*w0i
	im[i00] = cosT*i0 - sinT*w0r
	re[i01] = cosT*r1 + sinT*w1i
	im[i01] = cosT*i1 - sinT*w1r
	re[i10] = cosT*r2 + sinT*w2i
	im[i10] = cosT*i2 - sinT*w2r
	re[i11] = cosT*r3 + sinT*w3i
	im[i11] = cosT*i3 - sinT*w3r
}
