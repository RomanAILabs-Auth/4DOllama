package ga4

import "math/bits"

// Blade masks for orthonormal basis e0..e3 with e_k^2 = +1 (Cl(4,0)).
const (
	BladeScalar = uint8(0)
	BladeE0     = uint8(1 << 0)
	BladeE1     = uint8(1 << 1)
	BladeE2     = uint8(1 << 2)
	BladeE3     = uint8(1 << 3)
)

// gpBasis computes the geometric product of two basis blades (masks 0..15).
// Convention matches trailing-bit elimination against the left factor (metric all +1).
func gpBasis(a, b uint8) (mask uint8, sign float32) {
	aa := uint16(a)
	bb := uint16(b)
	s := float32(1)
	for bb != 0 {
		i := bits.TrailingZeros16(bb)
		bit := uint16(1) << i
		bb ^= bit
		low := aa & (bit - 1)
		if bits.OnesCount16(low)&1 == 1 {
			s = -s
		}
		if aa&bit != 0 {
			aa ^= bit
			s *= +1 // Cl(4,0) positive signature
		} else {
			aa ^= bit
		}
	}
	return uint8(aa), s
}
