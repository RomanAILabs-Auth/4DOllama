package clifford

import (
	"math"
	"testing"
)

func TestE1Sq(t *testing.T) {
	e1 := NewVector(1, 0, 0, 0)
	p := GeometricProduct(&e1, &e1)
	if math.Abs(p[0]-1) > 1e-14 || math.Abs(p[1]) > 1e-14 {
		t.Fatalf("e1^2 want scalar 1 got %+v", p)
	}
}

func TestAnticommute(t *testing.T) {
	e1 := NewVector(1, 0, 0, 0)
	e2 := NewVector(0, 1, 0, 0)
	a := GeometricProduct(&e1, &e2)
	b := GeometricProduct(&e2, &e1)
	for i := range a {
		if math.Abs(a[i]+b[i]) > 1e-12 {
			t.Fatalf("e1e2 + e2e1 != 0 at %d: %g %g", i, a[i], b[i])
		}
	}
}

func TestRotateXW(t *testing.T) {
	// Bivector e1∧e4 = blade product e1*e4
	R := RotorFromBivectorAngle(0b1001, math.Pi/2) // bits 0 and 3
	v := NewVector(1, 0, 0, 0)
	w := RotateVector(&R, &v)
	if math.Abs(math.Abs(w[8])-1) > 1e-10 || math.Abs(w[1]) > 1e-10 {
		t.Fatalf("90° x-w rotation expected w≈±1 got %+v", w)
	}
}
