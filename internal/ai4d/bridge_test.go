package ai4d

import (
	"math"
	"testing"
)

func TestFrobeniusCausalQKTRoPE_positive(t *testing.T) {
	rope := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	v := FrobeniusCausalQKTRoPE(rope)
	if v <= 0 || math.IsNaN(v) {
		t.Fatalf("expected positive finite norm, got %v", v)
	}
}

func TestIsoclinicRotorScale_bounded(t *testing.T) {
	s := IsoclinicRotorScale(0.5, 3)
	if s < 0.5 || s > 1.05 {
		t.Fatalf("scale out of range: %v", s)
	}
}
