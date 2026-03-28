package ga4

import "math"

// Rotor is an even-grade element R with R * ~R ≈ 1 used as v' = R v ~R on vectors.
// Stored as a full multivector for simplicity; callers should build plane rotors via FromBivectorPlane.
type Rotor struct {
	M Multivector
}

// IdentityRotor is the scalar 1.
func IdentityRotor() Rotor {
	var m Multivector
	m[0] = 1
	return Rotor{M: m}
}

// FromBivectorPlane builds R = cos(θ/2) + sin(θ/2) * B where B is the unit simple bivector e_i e_j (normalized by algebra).
// i,j are distinct basis indices 0..3. angleRad is the full rotation in radians.
func FromBivectorPlane(i, j int, angleRad float32) Rotor {
	if i == j || i < 0 || i > 3 || j < 0 || j > 3 {
		return IdentityRotor()
	}
	half := angleRad * 0.5
	ii, jj := uint8(1<<i), uint8(1<<j)
	mask, s := gpBasis(ii, jj)
	c := float32(math.Cos(float64(half)))
	sn := float32(math.Sin(float64(half))) * s
	var m Multivector
	m[0] = c
	m[mask] += sn
	return Rotor{M: m}
}

// RotateVec4 applies sandwich R v ~R; v is treated as sum of e_i.
func (r Rotor) RotateVec4(v Vec4) Vec4 {
	vm := v.AsMultivector()
	rev := r.M.Reverse()
	tmp := r.M.GeometricProduct(vm)
	out := tmp.GeometricProduct(rev)
	return Vec4FromGrade1(out)
}

// NormalizeEven approximates rotor normalization by scaling so scalar^2 + ||grade-2||^2 ≈ 1 (even subalgebra shortcut).
func (r Rotor) NormalizeEven() Rotor {
	var s2 float32
	for i := 0; i < 16; i++ {
		g := gradeOfBlade(uint8(i))
		if g == 0 || g == 2 || g == 4 {
			s2 += r.M[i] * r.M[i]
		}
	}
	if s2 <= 1e-20 {
		return IdentityRotor()
	}
	inv := float32(1) / float32(math.Sqrt(float64(s2)))
	return Rotor{M: r.M.Scale(inv)}
}
