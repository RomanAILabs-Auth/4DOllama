// Package ai4d bridges quaternion attention geometry to the 4D scalar lattice (runtime GGUF path).
package ai4d

import (
	"math"

	"github.com/4dollama/4dollama/internal/fourd/clifford"
)

const maxQuatSeqFrobenius = 64

func quatAt(buf []float32, tok int) [4]float32 {
	o := tok * 4
	if o+3 >= len(buf) {
		return [4]float32{1, 0, 0, 0}
	}
	return quatNormalize([4]float32{buf[o], buf[o+1], buf[o+2], buf[o+3]})
}

func quatNormalize(q [4]float32) [4]float32 {
	n := float32(math.Sqrt(float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])))
	if n <= 1e-6 {
		return [4]float32{1, 0, 0, 0}
	}
	return [4]float32{q[0] / n, q[1] / n, q[2] / n, q[3] / n}
}

func quatDot(a, b [4]float32) float32 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

// FrobeniusCausalQKTRoPE returns ||S||_F where S_ij = ⟨q_i, q_j⟩ over the causal block (j ≤ i),
// using RoPE quaternion rows from rope (length 4*seqLen). This matches the unscaled dot-product
// attention block produced when Q and K are both derived from the same RoPE stream.
func FrobeniusCausalQKTRoPE(rope []float32) float64 {
	if len(rope) < 4 {
		return 0
	}
	seqLen := len(rope) / 4
	if seqLen > maxQuatSeqFrobenius {
		rope = rope[len(rope)-maxQuatSeqFrobenius*4:]
		seqLen = maxQuatSeqFrobenius
	}
	var sumSq float64
	nTerms := 0
	for i := 0; i < seqLen; i++ {
		qi := quatAt(rope, i)
		for j := 0; j <= i; j++ {
			qj := quatAt(rope, j)
			d := float64(quatDot(qi, qj))
			sumSq += d * d
			nTerms++
		}
	}
	if nTerms == 0 {
		return 0
	}
	return math.Sqrt(sumSq / float64(nTerms))
}

// CognitiveGravityFromQKTFrobenius maps ||QKᵀ||_F (or RMS causal block norm) into lattice injection scale.
func CognitiveGravityFromQKTFrobenius(frob float64, kappa float64) float64 {
	if frob <= 0 || kappa <= 0 {
		return 0
	}
	g := frob * kappa * 1.2
	if g > 1e6 {
		g = 1e6
	}
	return g
}

// IsoclinicRotorScale builds R = R_xy(θ)·R_zw(θ) with θ tied to cognitive gravity and returns a
// stable positive scale for reverse coupling (scalar part magnitude, clamped).
func IsoclinicRotorScale(gravity float64, step int) float64 {
	theta := math.Atan(gravity) * 0.08
	if step > 0 {
		theta += 0.012 * math.Sin(float64(step)*0.37)
	}
	R := clifford.IsoclinicRotor(theta)
	s := math.Abs(R[0])
	if s < 0.15 {
		s = 0.15
	}
	if s > 1.0 {
		s = 1.0
	}
	return 0.55 + 0.45*s
}
