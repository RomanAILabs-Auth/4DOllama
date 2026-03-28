package compiler

import (
	"fmt"
	"io"
	"time"
)

// BuildPhase is one timed slice of the compile pipeline.
type BuildPhase struct {
	Name string
	D    time.Duration
}

// BuildBench accumulates per-phase timings. All methods are nil-safe.
type BuildBench struct {
	Phases []BuildPhase
}

// Add records one pipeline phase. Safe to call with nil receiver.
func (b *BuildBench) Add(name string, d time.Duration) {
	if b == nil {
		return
	}
	b.Phases = append(b.Phases, BuildPhase{Name: name, D: d})
}

// WriteReport prints a human-readable table to w (typically stderr).
func (b *BuildBench) WriteReport(w io.Writer, sourceLabel string) {
	if b == nil || len(b.Phases) == 0 {
		return
	}
	var total time.Duration
	for _, p := range b.Phases {
		total += p.D
	}
	fmt.Fprintf(w, "r4d bench - %s\n", sourceLabel)
	for _, p := range b.Phases {
		fmt.Fprintf(w, "  %-26s %10.3f ms\n", p.Name+":", ms(p.D))
	}
	fmt.Fprintf(w, "  %-26s %10.3f ms\n", "total (sum of phases):", ms(total))
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}
