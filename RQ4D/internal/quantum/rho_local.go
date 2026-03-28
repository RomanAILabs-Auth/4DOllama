package quantum

import (
	"math"
	"sync"
)

// applyZPhaseRho applies exp(-i dt H_Z) on the left/right of ρ (diagonal unitary sandwich).
func applyZPhaseRho(rhoRe, rhoIm []float64, d int, hz []float64, nq int, dt float64) {
	for a := 0; a < d; a++ {
		pha := sumZexpect(a, hz, nq)
		for b := 0; b < d; b++ {
			phb := sumZexpect(b, hz, nq)
			phi := -dt * (pha - phb)
			c, s := math.Cos(phi), math.Sin(phi)
			rr, ri := rhoRe[a*d+b], rhoIm[a*d+b]
			rhoRe[a*d+b] = rr*c - ri*s
			rhoIm[a*d+b] = rr*s + ri*c
		}
	}
}

// rVal returns R_{pa} for the single-qubit X-rotation embedding on (i0,i1).
func rVal(p, a, i0, i1 int, cosA, sinA float64) (float64, float64) {
	w00r, w00i := cosA, 0.0
	w01r, w01i := 0.0, -sinA
	w10r, w10i := 0.0, -sinA
	w11r, w11i := cosA, 0.0
	if p != i0 && p != i1 {
		if a == p {
			return 1, 0
		}
		return 0, 0
	}
	if p == i0 {
		switch a {
		case i0:
			return w00r, w00i
		case i1:
			return w01r, w01i
		default:
			return 0, 0
		}
	}
	switch a {
	case i0:
		return w10r, w10i
	case i1:
		return w11r, w11i
	default:
		return 0, 0
	}
}

// rhoSandwichOneXPair applies ρ' = R ρ R† for one X-rotation on qubit line (i0,i1).
func rhoSandwichOneXPair(rhoRe, rhoIm []float64, d, i0, i1 int, cosA, sinA float64, outRe, outIm []float64) {
	for p := 0; p < d; p++ {
		for q := 0; q < d; q++ {
			var sumR, sumI float64
			for a := 0; a < d; a++ {
				for b := 0; b < d; b++ {
					rpaR, rpaI := rVal(p, a, i0, i1, cosA, sinA)
					rqbR, rqbI := rVal(q, b, i0, i1, cosA, sinA)
					rar, rai := rhoRe[a*d+b], rhoIm[a*d+b]
					tr := rpaR*rar - rpaI*rai
					ti := rpaR*rai + rpaI*rar
					// multiply by conj(R_qb)
					sumR += tr*rqbR + ti*rqbI
					sumI += ti*rqbR - tr*rqbI
				}
			}
			outRe[p*d+q] = sumR
			outIm[p*d+q] = sumI
		}
	}
}

func applyPauliXLayerRho(rhoRe, rhoIm []float64, d, k int, angle float64, tmpRe, tmpIm []float64) {
	stride := 1 << k
	cosA, sinA := math.Cos(angle), math.Sin(angle)
	for b0 := 0; b0 < d; b0 += 2 * stride {
		for off := 0; off < stride; off++ {
			i0 := b0 + off
			i1 := b0 + off + stride
			rhoSandwichOneXPair(rhoRe, rhoIm, d, i0, i1, cosA, sinA, tmpRe, tmpIm)
			copy(rhoRe[:d*d], tmpRe[:d*d])
			copy(rhoIm[:d*d], tmpIm[:d*d])
		}
	}
}

// applyLocalStrangRhoParallel applies the same Strang splitting as ψ on each site's ρ.
// srcFlat and dstFlat each hold [real block][imag block] per site in one slice.
func (S *Simulator) applyLocalStrangRhoParallel(srcFlat, dstFlat []float64, workers int) {
	d := S.Dim
	n := S.N
	dd := d * d
	chunk := (n + workers - 1) / workers
	var wg2 sync.WaitGroup
	for t := 0; t < workers; t++ {
		i0 := t * chunk
		i1 := i0 + chunk
		if i1 > n {
			i1 = n
		}
		if i0 >= i1 {
			break
		}
		wg2.Add(1)
		go func(a, b int) {
			defer wg2.Done()
			var tmpRe, tmpIm [maxDim * maxDim]float64
			for s := a; s < b; s++ {
				or := rhoOffset(s, d)
				copy(dstFlat[or:or+2*dd], srcFlat[or:or+2*dd])
				br, bi := rhoPtrFlat(dstFlat, s, d)
				applyZPhaseRho(br, bi, d, S.Hz, S.NQ, S.Dt*0.5)
				for k := 0; k < S.NQ; k++ {
					applyPauliXLayerRho(br, bi, d, k, -S.Dt*S.Hx[k], tmpRe[:], tmpIm[:])
				}
				applyZPhaseRho(br, bi, d, S.Hz, S.NQ, S.Dt*0.5)
			}
		}(i0, i1)
	}
	wg2.Wait()
}
