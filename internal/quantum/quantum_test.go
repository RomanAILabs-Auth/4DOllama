package quantum

import (
	"encoding/hex"
	"math"
	"testing"
)

func TestGlobalNormPreserved_LocalOnlyCoupling(t *testing.T) {
	// JBond = 0 ⇒ bond XX is identity; Schmidt still refactors product state (rank-1 exact).
	S := NewSimulator(4, 4, 4, 2, 2)
	S.Dt = 0.03
	S.JBond = 0
	S.SetUniformFields(0.1, 0.12)
	S.InitProductComputational()
	S.NormalizeGlobal(S.ReA, S.ImA)
	n0 := S.GlobalNorm2(S.ReA, S.ImA)
	for i := 0; i < 8; i++ {
		S.Step()
	}
	n1 := S.GlobalNorm2(S.ReA, S.ImA)
	if math.Abs(n0-1) > 1e-10 || math.Abs(n1-1) > 1e-4 {
		t.Fatalf("norm drift: before=%g after=%g (want ~1)", n0, n1)
	}
}

func TestDeterminismHashWorkers(t *testing.T) {
	run := func(workers int) [32]byte {
		S := NewSimulator(3, 3, 3, 2, workers)
		S.Dt = 0.04
		S.JBond = 0.05
		S.SetUniformFields(0.07, 0.06)
		for i := range S.ReA {
			S.ReA[i] = 0.01 * float64(i%7)
			S.ImA[i] = 0.02 * float64((i+3)%5)
		}
		S.NormalizeGlobal(S.ReA, S.ImA)
		for i := 0; i < 5; i++ {
			S.Step()
		}
		return S.StateHashSHA256()
	}
	h1 := run(1)
	h4 := run(4)
	if h1 != h4 {
		t.Fatalf("hash mismatch:\n%s\n%s", hex.EncodeToString(h1[:]), hex.EncodeToString(h4[:]))
	}
}

func TestUnitaryLocalPreservesSiteNormSum(t *testing.T) {
	S := NewSimulator(2, 2, 2, 2, 2)
	S.Dt = 0.1
	S.JBond = 0
	S.SetUniformFields(0.3, 0.2)
	for i := range S.ReA {
		S.ReA[i] = float64((i*i)%3) * 0.2
		S.ImA[i] = float64((i+1)%2) * 0.15
	}
	S.NormalizeGlobal(S.ReA, S.ImA)
	var sum float64
	for s := 0; s < S.N; s++ {
		sum += S.SiteNorm(s)
	}
	for i := 0; i < 10; i++ {
		S.Step()
	}
	var sum2 float64
	for s := 0; s < S.N; s++ {
		sum2 += S.SiteNorm(s)
	}
	if math.Abs(sum-sum2) > 1e-3 {
		t.Fatalf("site norm sum changed: %g vs %g", sum, sum2)
	}
}

func TestMeasureProbabilities(t *testing.T) {
	S := NewSimulator(1, 1, 1, 2, 1)
	S.ReA[0] = 1 / math.Sqrt(2)
	S.ImA[0] = 0
	S.ReA[1] = 1 / math.Sqrt(2)
	S.ImA[1] = 0
	p0 := S.ProbK(0, 0)
	p1 := S.ProbK(0, 1)
	if math.Abs(p0-0.5) > 1e-9 || math.Abs(p1-0.5) > 1e-9 {
		t.Fatalf("p0=%g p1=%g", p0, p1)
	}
}

func TestTNChi1MatchesMeanFieldPsiEvolution(t *testing.T) {
	init := func(S *Simulator) {
		S.Dt = 0.04
		S.JBond = 0.05
		S.SetUniformFields(0.07, 0.06)
		for i := range S.ReA {
			S.ReA[i] = 0.01 * float64(i%7)
			S.ImA[i] = 0.02 * float64((i+3)%5)
		}
		S.NormalizeGlobal(S.ReA, S.ImA)
	}
	mf := NewSimulator(3, 3, 3, 2, 2)
	init(mf)
	tn := NewSimulator(3, 3, 3, 2, 2)
	tn.SetQuantumBackend(BackendTN, 1)
	init(tn)
	tn.InitRhoComputational()
	tn.SyncAllRhoFromPsi(tn.ReA, tn.ImA)
	copy(tn.RhoB, tn.RhoA)

	for i := 0; i < 4; i++ {
		mf.Step()
		tn.Step()
	}
	if math.Abs(mf.GlobalNorm2(mf.ReA, mf.ImA)-tn.GlobalNorm2(tn.ReA, tn.ImA)) > 1e-8 {
		t.Fatalf("psi global norm mismatch")
	}
	if math.Abs(tn.GlobalTraceRho()-float64(tn.N)) > 1e-2 {
		t.Fatalf("want Tr rho sum ~ N, got %g", tn.GlobalTraceRho())
	}
}

func TestTNMultDeterminismWorkers(t *testing.T) {
	run := func(workers int) [32]byte {
		S := NewSimulator(3, 3, 3, 2, workers)
		S.SetQuantumBackend(BackendTN, 2)
		S.Dt = 0.03
		S.JBond = 0.04
		S.SetUniformFields(0.05, 0.05)
		S.InitProductComputational()
		S.NormalizeGlobal(S.ReA, S.ImA)
		S.SyncAllRhoFromPsi(S.ReA, S.ImA)
		copy(S.RhoB, S.RhoA)
		for i := 0; i < 3; i++ {
			S.Step()
		}
		return S.StateHashSHA256()
	}
	a := run(1)
	b := run(4)
	if a != b {
		t.Fatalf("TN χ>1 hash mismatch across workers")
	}
}

func TestMeasureDeterministicWithSeed(t *testing.T) {
	S := NewSimulator(1, 1, 1, 2, 1)
	S.ReA[0] = 0.6
	S.ReA[1] = 0.8
	S.ImA[0], S.ImA[1] = 0, 0
	S.NormalizeGlobal(S.ReA, S.ImA)
	r1 := NewRNG(999)
	r2 := NewRNG(999)
	a := S.MeasureSite(0, r1, false)
	b := S.MeasureSite(0, r2, false)
	if a != b {
		t.Fatalf("same seed different outcome: %d vs %d", a, b)
	}
}
