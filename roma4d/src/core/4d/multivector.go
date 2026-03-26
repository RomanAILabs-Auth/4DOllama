package ga4

// Multivector is an element of Cl(4,0) in the 16-dimensional basis.
// Index i stores the coefficient of basis blade with bitmask i (0..15).
type Multivector [16]float32

// GeometricProduct is the full bilinear product on Cl(4,0).
func (a Multivector) GeometricProduct(b Multivector) (out Multivector) {
	for i := 0; i < 16; i++ {
		if a[i] == 0 {
			continue
		}
		for j := 0; j < 16; j++ {
			if b[j] == 0 {
				continue
			}
			k, s := gpBasis(uint8(i), uint8(j))
			out[k] += a[i] * b[j] * s
		}
	}
	return out
}

// Add returns a + b.
func (a Multivector) Add(b Multivector) (out Multivector) {
	for i := 0; i < 16; i++ {
		out[i] = a[i] + b[i]
	}
	return out
}

// Scale returns s * a.
func (a Multivector) Scale(s float32) (out Multivector) {
	for i := 0; i < 16; i++ {
		out[i] = a[i] * s
	}
	return out
}

// Reverse applies the grade involution's conjugate (reverse) on basis blades.
func (a Multivector) Reverse() (out Multivector) {
	// Grade k picks up (-1)^(k(k-1)/2)
	for i := 0; i < 16; i++ {
		g := gradeOfBlade(uint8(i))
		if (g*(g-1)/2)&1 == 1 {
			out[i] = -a[i]
		} else {
			out[i] = a[i]
		}
	}
	return out
}

func gradeOfBlade(m uint8) int {
	return popCount4(m)
}

func popCount4(m uint8) int {
	n := 0
	for m != 0 {
		n++
		m &= m - 1
	}
	return n
}
