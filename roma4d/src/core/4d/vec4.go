package ga4

// Vec4 is a Euclidean 4-vector (grade-1 part in Cl(4,0) basis e0..e3).
type Vec4 struct {
	X, Y, Z, W float32
}

func (v Vec4) Add(u Vec4) Vec4 {
	return Vec4{v.X + u.X, v.Y + u.Y, v.Z + u.Z, v.W + u.W}
}

func (v Vec4) Scale(s float32) Vec4 {
	return Vec4{v.X * s, v.Y * s, v.Z * s, v.W * s}
}

// AsMultivector embeds v along basis e0..e3 (blade bitmasks 1,2,4,8).
func (v Vec4) AsMultivector() Multivector {
	var m Multivector
	m[BladeE0] = v.X
	m[BladeE1] = v.Y
	m[BladeE2] = v.Z
	m[BladeE3] = v.W
	return m
}

// Vec4FromGrade1 reads grade-1 coefficients from m.
func Vec4FromGrade1(m Multivector) Vec4 {
	return Vec4{m[BladeE0], m[BladeE1], m[BladeE2], m[BladeE3]}
}
