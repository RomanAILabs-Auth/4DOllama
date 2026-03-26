package ga4

import (
	"math"
	"testing"
)

func TestGpBasis_ScalarSquare(t *testing.T) {
	m, s := gpBasis(BladeE0, BladeE0)
	if m != BladeScalar || math.Abs(float64(s-1)) > 1e-6 {
		t.Fatalf("e0*e0 want scalar +1, got mask=%d sign=%v", m, s)
	}
}

func TestGeometricProduct_VecNorm(t *testing.T) {
	v := Vec4{1, 0, 0, 0}.AsMultivector()
	n2 := v.GeometricProduct(v)
	if math.Abs(float64(n2[0]-1)) > 1e-5 {
		t.Fatalf("e0^2 expected +1 scalar, got %v", n2[0])
	}
}

func TestRotor_PreservesRoughNorm(t *testing.T) {
	r := FromBivectorPlane(0, 1, float32(math.Pi/4)).NormalizeEven()
	v := Vec4{1, 1, 0, 0}
	out := r.RotateVec4(v)
	// Sandwich with normalized rotor should keep vector-like magnitude roughly stable
	inN := float64(v.X*v.X + v.Y*v.Y + v.Z*v.Z + v.W*v.W)
	outN := float64(out.X*out.X + out.Y*out.Y + out.Z*out.Z + out.W*out.W)
	if math.Abs(inN-outN) > 1e-3*inN {
		t.Fatalf("norm drift: in=%v out=%v", inN, outN)
	}
}
