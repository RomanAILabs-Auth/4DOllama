package cli

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/4dollama/4dollama/internal/fourd/clifford"
	"github.com/4dollama/4dollama/internal/fourd/orchestrator"
)

func cmdFourd(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprint(os.Stderr, `usage: 4dollama fourd <subcommand>

Subcommands:
  lattice   Run 4D wave + QKT coupling orchestrator (native Go numerical core)
  ga-demo   Print Cl(4,0) rotor / isoclinic self-test to stdout

Roma4D (.r4d) is the primary *language* for shipped 4D kernels; this subcommand
runs the 4DOllama numerical substrate inside the 4dollama binary.
`)
		return 2
	}
	switch args[0] {
	case "lattice":
		return cmdFourdLattice(args[1:], log)
	case "ga-demo":
		return cmdFourdGADemo()
	default:
		fmt.Fprintf(os.Stderr, "4dollama fourd: unknown subcommand %q\n", args[0])
		return 2
	}
}

func cmdFourdLattice(args []string, log *slog.Logger) int {
	fs := flag.NewFlagSet("fourd lattice", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	steps := fs.Int("steps", 120, "orchestrator steps")
	kappa := fs.Float64("kappa", 0.002, "cognitive gravity scale (||QK^T||_F multiplier)")
	inject := fs.Int("inject-every", 4, "inject QKT source every N steps (0=off)")
	_ = fs.Parse(args)

	cfg := orchestrator.LoopConfig{
		Steps:        *steps,
		Tick:         0,
		WaveC:        0.12,
		WaveDt:       0.15,
		InjectEvery:  *inject,
		GravityKappa: *kappa,
		SmoothIters:  0,
		SmoothAlpha:  0,
	}
	ctx := context.Background()
	if log == nil {
		log = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	}
	if err := orchestrator.RunLatticeLoop(ctx, log, os.Stdout, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "fourd lattice: %v\n", err)
		return 1
	}
	return 0
}

func cmdFourdGADemo() int {
	th := 0.7
	R := clifford.IsoclinicRotor(th)
	v := clifford.NewVector(1, 0, 1, 0)
	w := clifford.RotateVector(&R, &v)
	fmt.Printf("4DOllama Cl(4,0) demo\n")
	fmt.Printf("  isoclinic θ=%.3f rotor grade-0 coeff R[0]=%.4f\n", th, R[0])
	fmt.Printf("  vector (1,0,1,0) -> (%.4f, %.4f, %.4f, %.4f)\n", w[1], w[2], w[4], w[8])
	fmt.Printf("  (Roma4D .r4d sources compile rotors to native LLVM; this is the in-process check.)\n")
	return 0
}
