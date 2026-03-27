// Command rq4d runs the quantum lattice simulator (see package quantum).
package main

import (
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/4dollama/4dollama/internal/quantum"
)

func main() {
	var (
		lx       = flag.Int("lx", 16, "lattice extent x")
		ly       = flag.Int("ly", 16, "lattice extent y")
		lz       = flag.Int("lz", 16, "lattice extent z")
		dim      = flag.Int("dim", 2, "local Hilbert dimension: 2, 4, or 8")
		dt       = flag.Float64("dt", 0.05, "Trotter timestep Δt")
		steps    = flag.Int("steps", 20, "number of Step() calls")
		seed     = flag.Int64("seed", 42, "PRNG seed (measurements / sampling only)")
		workers  = flag.Int("workers", 0, "goroutine workers (0 = GOMAXPROCS)")
		jbond    = flag.Float64("j", 0.3, "XX bond strength J in H = J Σ X_k X_k'")
		hz0      = flag.Float64("hz", 0.2, "uniform on-site Z coefficient")
		hx0      = flag.Float64("hx", 0.15, "uniform on-site X coefficient")
		measure  = flag.Bool("measure", false, "after evolution, run destructive measurement on site 0")
		collapse = flag.Bool("collapse", true, "when measuring, collapse state (destructive)")
		backend  = flag.String("backend", "meanfield", "quantum backend: meanfield | tn | cpu (cpu=meanfield)")
		chi      = flag.Int("chi", 1, "bond dimension χ for --backend=tn (1≈product; 2–32 entanglement)")
	)
	flag.Parse()
	if *lx < 1 || *ly < 1 || *lz < 1 {
		fmt.Fprintln(os.Stderr, "lx, ly, lz must be >= 1")
		os.Exit(2)
	}
	if *dim != 2 && *dim != 4 && *dim != 8 {
		fmt.Fprintln(os.Stderr, "dim must be 2, 4, or 8")
		os.Exit(2)
	}
	if *backend != "meanfield" && *backend != "tn" && *backend != "cpu" {
		fmt.Fprintf(os.Stderr, "backend must be meanfield, tn, or cpu (got %q)\n", *backend)
		os.Exit(2)
	}
	if *chi < 1 || *chi > quantum.MaxChiCap {
		fmt.Fprintf(os.Stderr, "chi must be in [1,%d]\n", quantum.MaxChiCap)
		os.Exit(2)
	}

	n := int64(*lx) * int64(*ly) * int64(*lz)
	d := int64(*dim)
	mem := 2 * n * d * 2 * 8 // two ψ buffers (re+im float64)
	if *backend == "tn" {
		mem += 2 * n * d * d * 2 * 8 // two ρ buffers per site (Hermitian d×d, re+im)
	}
	fmt.Printf("nodes=%d dim=%d backend=%s chi=%d memory_est_B=%d\n", n, *dim, *backend, *chi, mem)

	w := *workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}

	S := quantum.NewSimulator(*lx, *ly, *lz, *dim, w)
	S.SetQuantumBackend(*backend, *chi)
	S.Dt = *dt
	S.JBond = *jbond
	S.SetUniformFields(*hz0, *hx0)
	S.Workers = w

	S.InitProductComputational()
	S.NormalizeGlobal(S.ReA, S.ImA)
	if S.Backend == quantum.BackendTN {
		S.SyncAllRhoFromPsi(S.ReA, S.ImA)
		copy(S.RhoB, S.RhoA)
	}

	n0 := S.GlobalNorm2(S.ReA, S.ImA)
	for s := 0; s < *steps; s++ {
		S.Step()
	}
	n1 := S.GlobalNorm2(S.ReA, S.ImA)
	h := S.StateHashSHA256()
	eh := S.ExpectationH()

	fmt.Printf("go_maxprocs=%d workers=%d\n", runtime.GOMAXPROCS(0), w)
	fmt.Printf("global_norm2_before=%.12g after=%.12g\n", n0, n1)
	fmt.Printf("expectation_H_mf=%.12g\n", eh)
	fmt.Printf("state_sha256=%s\n", hex.EncodeToString(h[:]))

	if *measure {
		rng := quantum.NewRNG(*seed)
		k := S.MeasureSite(0, rng, *collapse)
		fmt.Printf("measure_site0_outcome=%d\n", k)
		h2 := S.StateHashSHA256()
		fmt.Printf("state_sha256_after_measure=%s\n", hex.EncodeToString(h2[:]))
	}
}
