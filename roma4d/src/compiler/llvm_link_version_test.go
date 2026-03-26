package compiler

import "testing"

func TestGccVersionLess(t *testing.T) {
	tests := []struct {
		a, b string
		want bool // want gccVersionLess(a,b)
	}{
		{"13.2.0", "14.1.0", true},
		{"14.1.0", "13.2.0", false},
		{"13.2.0", "13.2.0", false},
		{"9.0.0", "10.0.0", true},
		{"10.0.0", "9.0.0", false},
	}
	for _, tc := range tests {
		if got := gccVersionLess(tc.a, tc.b); got != tc.want {
			t.Fatalf("gccVersionLess(%q,%q)=%v want %v", tc.a, tc.b, got, tc.want)
		}
	}
}
