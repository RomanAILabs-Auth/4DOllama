package ai4d

import "math"

// Vec4 is a column in R^4 (w,x,y,z) used for Cl(4,0) minimal realizations.
type Vec4 [4]float64

// Mat4 stores a 4×4 matrix in row-major order (row i, col j at [i*4+j]).
type Mat4 [16]float64

// Mul4 applies m to column v (out-of-place).
func Mul4(m Mat4, v Vec4) Vec4 {
	var o Vec4
	for i := 0; i < 4; i++ {
		s := 0.0
		for j := 0; j < 4; j++ {
			s += m[i*4+j] * v[j]
		}
		o[i] = s
	}
	return o
}

// Quat is (w, x, y, z) with real part first, Hamilton product convention.
type Quat Vec4

// QuatMul returns a * b.
func QuatMul(a, b Quat) Quat {
	return Quat{
		a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
		a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
		a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
		a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
	}
}

// QuatConj returns quaternion conjugate.
func QuatConj(q Quat) Quat {
	return Quat{q[0], -q[1], -q[2], -q[3]}
}

// QuatNorm returns Euclidean norm.
func QuatNorm(q Quat) float64 {
	return math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
}

// QuatNormalize returns q/||q||; if zero, identity.
func QuatNormalize(q Quat) Quat {
	n := QuatNorm(q)
	if n < 1e-15 {
		return Quat{1, 0, 0, 0}
	}
	return Quat{q[0] / n, q[1] / n, q[2] / n, q[3] / n}
}

// DoubleIsoclinic applies the standard (S³×S³)/{±1} action on R^4 ≅ ℍ:
// v ↦ qL * v * conj(qR), with v embedded as pure quaternion (0, v0, v1, v2) if we use first three;
// for full R^4 use v as quaternion (v0,v1,v2,v3).
func DoubleIsoclinic(v Vec4, qL, qR Quat) Vec4 {
	ql := QuatNormalize(qL)
	qr := QuatNormalize(qR)
	vq := Quat{v[0], v[1], v[2], v[3]}
	t := QuatMul(QuatMul(ql, vq), QuatConj(qr))
	return Vec4{t[0], t[1], t[2], t[3]}
}

// RotorInPlane builds unit quaternions for an isoclinic rotation in the given bivector plane.
// plane: "XY", "XZ", "YZ", "XW", "YW", "ZW" (W is fourth axis v[3]).
func RotorInPlane(plane string, angle float64) (qL, qR Quat) {
	h := angle * 0.5
	c, s := math.Cos(h), math.Sin(h)
	switch plane {
	case "XY", "xy":
		// e1e2-like: left rotor exp(-½ θ e12)
		qL = QuatNormalize(Quat{c, 0, 0, s})
		qR = Quat{c, 0, 0, s}
	case "XZ", "xz":
		qL = QuatNormalize(Quat{c, 0, -s, 0})
		qR = Quat{c, 0, -s, 0}
	case "YZ", "yz":
		qL = QuatNormalize(Quat{c, s, 0, 0})
		qR = Quat{c, s, 0, 0}
	case "XW", "xw":
		qL = QuatNormalize(Quat{c, -s, 0, 0})
		qR = QuatNormalize(Quat{c, s, 0, 0})
	case "YW", "yw":
		qL = QuatNormalize(Quat{c, 0, s, 0})
		qR = QuatNormalize(Quat{c, 0, -s, 0})
	case "ZW", "zw":
		qL = QuatNormalize(Quat{c, 0, 0, -s})
		qR = QuatNormalize(Quat{c, 0, 0, s})
	default:
		qL = Quat{1, 0, 0, 0}
		qR = Quat{1, 0, 0, 0}
	}
	return qL, qR
}

// ApplyCliffordTransform applies linear 4×4 then double-isoclinic rotation (deterministic).
func ApplyCliffordTransform(v Vec4, linear Mat4, plane string, angle float64) Vec4 {
	u := Mul4(linear, v)
	qL, qR := RotorInPlane(plane, angle)
	return DoubleIsoclinic(u, qL, qR)
}

// Mat4Mul returns a * b (row-major 4×4).
func Mat4Mul(a, b Mat4) Mat4 {
	var c Mat4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			s := 0.0
			for k := 0; k < 4; k++ {
				s += a[i*4+k] * b[k*4+j]
			}
			c[i*4+j] = s
		}
	}
	return c
}

func planeAxes(plane string) (i, j int, ok bool) {
	switch plane {
	case "XY", "xy":
		return 0, 1, true
	case "XZ", "xz":
		return 0, 2, true
	case "YZ", "yz":
		return 1, 2, true
	case "XW", "xw":
		return 0, 3, true
	case "YW", "yw":
		return 1, 3, true
	case "ZW", "zw":
		return 2, 3, true
	default:
		return 0, 0, false
	}
}

func identity4() Mat4 {
	return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

// Mat4RotatePlane returns a right-handed rotation in the given coordinate plane (axes 0..3 = v0..v3).
func Mat4RotatePlane(plane string, angle float64) Mat4 {
	R := identity4()
	i, j, ok := planeAxes(plane)
	if !ok {
		return R
	}
	c, s := math.Cos(angle), math.Sin(angle)
	R[i*4+i] = c
	R[i*4+j] = -s
	R[j*4+i] = s
	R[j*4+j] = c
	return R
}
