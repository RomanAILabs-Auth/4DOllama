package lattice4

import "testing"

func TestHeatConservesMeanApprox(t *testing.T) {
	a := NewGrid(8, 8, 8, 4)
	b := NewGrid(8, 8, 8, 4)
	a.Set(3, 3, 3, 2, 1)
	sum0 := 0.0
	for _, v := range a.Data {
		sum0 += v
	}
	StepHeat(a, b, nil, 0.1, 0.05)
	sum1 := 0.0
	for _, v := range b.Data {
		sum1 += v
	}
	if sum1 < sum0*0.5 || sum1 > sum0*1.5 {
		t.Fatalf("unexpected mass change sum0=%v sum1=%v", sum0, sum1)
	}
}
