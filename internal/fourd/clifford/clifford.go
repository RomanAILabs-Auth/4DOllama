// Package clifford implements Cl(4,0) multivectors (16 basis blades, metric ++++).
// Basis index bit i corresponds to generator e_{i+1}. This is the numerical substrate
// for 4DOllama native geometry; Roma4D .r4d sources compile to similar ops via LLVM.
package clifford

import "math"

const Dim = 16

// Multivector holds 16 float64 coefficients in blade-bit order (index = bitmask of {e1,e2,e3,e4}).
type Multivector [Dim]float64

// NewVector is the grade-1 point P = x e1 + y e2 + z e3 + w e4.
func NewVector(x, y, z, w float64) Multivector {
	var m Multivector
	m[1] = x // e1
	m[2] = y // e2
	m[4] = z // e3
	m[8] = w // e4
	return m
}

// orderedBladeProduct combines basis blades a,b into basis blade with sign (Cl(4,0), all e_i^2=+1).
func orderedBladeProduct(a, b uint8) (uint8, float64) {
	var s []uint8
	for k := uint8(0); k < 4; k++ {
		if a&(1<<k) != 0 {
			s = append(s, k)
		}
	}
	for k := uint8(0); k < 4; k++ {
		if b&(1<<k) != 0 {
			s = append(s, k)
		}
	}
	sign := 1.0
	for {
		changed := false
		for i := 0; i < len(s)-1; i++ {
			if s[i] > s[i+1] {
				s[i], s[i+1] = s[i+1], s[i]
				sign = -sign
				changed = true
			}
		}
		merged := false
		for i := 0; i < len(s)-1; i++ {
			if s[i] == s[i+1] {
				s = append(s[:i], s[i+2:]...)
				merged = true
				changed = true
				break
			}
		}
		if !changed && !merged {
			break
		}
		if merged {
			continue
		}
		if !changed {
			break
		}
	}
	var out uint8
	for _, k := range s {
		out |= 1 << k
	}
	return out, sign
}

// GeometricProduct computes the full Clifford product (distributive over basis).
func GeometricProduct(a, b *Multivector) (out Multivector) {
	var o Multivector
	for i := 0; i < Dim; i++ {
		if a[i] == 0 {
			continue
		}
		for j := 0; j < Dim; j++ {
			if b[j] == 0 {
				continue
			}
			k, sgn := orderedBladeProduct(uint8(i), uint8(j))
			o[k] += sgn * a[i] * b[j]
		}
	}
	return o
}

// Reverse reverses blade order (sign = (-1)^(k(k-1)/2) on grade k).
func Reverse(m *Multivector) (out Multivector) {
	for i := 0; i < Dim; i++ {
		k := popcount(uint8(i))
		sign := 1.0
		if (k*(k-1)/2)%2 == 1 {
			sign = -1.0
		}
		out[i] = sign * m[i]
	}
	return out
}

func popcount(x uint8) int {
	n := 0
	for x != 0 {
		n++
		x &= x - 1
	}
	return n
}

// RotorPlaneBivectorMask returns the blade index for e_i ∧ e_j (i,j in 0..3 for e1..e4).
func RotorPlaneBivectorMask(i, j int) int {
	if i == j || i < 0 || j < 0 || i > 3 || j > 3 {
		return -1
	}
	bi := uint8(1 << i)
	bj := uint8(1 << j)
	k, sgn := orderedBladeProduct(bi, bj)
	if sgn < 0 {
		return int(k) | 0 // keep sign in separate API
	}
	return int(k)
}

// RotorFromBivectorAngle builds R = cos(θ/2) + sin(θ/2)*B for unit bivector blade B at mask (odd orientation may flip sign).
// B must be a single basis bivector (one blade index with grade 2).
func RotorFromBivectorAngle(bivectorBlade int, theta float64) Multivector {
	var R Multivector
	c := math.Cos(theta / 2)
	s := math.Sin(theta / 2)
	R[0] = c
	if bivectorBlade > 0 && bivectorBlade < Dim {
		R[bivectorBlade] = s
	}
	return R
}

// Sandwich applies R * M * ~R.
func Sandwich(R, M *Multivector) Multivector {
	Rrev := Reverse(R)
	tmp := GeometricProduct(R, M)
	return GeometricProduct(&tmp, &Rrev)
}

// RotateVector applies sandwich with rotor R to a grade-1 vector (other grades truncated in result extraction).
func RotateVector(R *Multivector, v *Multivector) Multivector {
	out := Sandwich(R, v)
	var w Multivector
	w[1] = out[1]
	w[2] = out[2]
	w[4] = out[4]
	w[8] = out[8]
	return w
}

// IsoclinicRotor returns R = R_xy(θ) * R_zw(θ) as geometric product (simultaneous equal-angle double rotation).
func IsoclinicRotor(theta float64) Multivector {
	// e1e2 blade index = 0b0011 = 3, e3e4 = 0b1100 = 12
	Rxy := RotorFromBivectorAngle(0b0011, theta)
	Rzw := RotorFromBivectorAngle(0b1100, theta)
	return GeometricProduct(&Rxy, &Rzw)
}
