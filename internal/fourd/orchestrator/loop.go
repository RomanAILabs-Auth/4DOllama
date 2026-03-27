// Package orchestrator runs bounded research loops (logging, stepping, optional export hooks).
package orchestrator

import (
	"context"
	"io"
	"log/slog"
	"time"

	"github.com/4dollama/4dollama/internal/fourd/coupling"
	"github.com/4dollama/4dollama/internal/fourd/hodge"
	"github.com/4dollama/4dollama/internal/fourd/lattice4"
)

// LoopConfig drives a simple persistent simulation loop.
type LoopConfig struct {
	Steps           int
	Tick            time.Duration
	WaveC, WaveDt   float64
	InjectEvery     int
	GravityKappa    float64
	SmoothIters     int
	SmoothAlpha     float64
}

// RunLatticeLoop executes wave + optional Q-proxy injection + harmonic proxy each frame.
func RunLatticeLoop(ctx context.Context, log *slog.Logger, w io.Writer, cfg LoopConfig) error {
	g := lattice4.NewGrid(12, 12, 12, 8)
	g.Set(6, 6, 6, 4, 1.0) // impulse
	cur := lattice4.NewGrid(12, 12, 12, 8)
	copy(cur.Data, g.Data)
	nxt := lattice4.NewGrid(12, 12, 12, 8)
	src := lattice4.NewGrid(12, 12, 12, 8)
	qwork := make([]float64, 16*16)
	D := cfg.WaveC * cfg.WaveC // interpret as diffusion coefficient scale (stable PDE path)

	for step := 0; step < cfg.Steps; step++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		clear(src.Data)
		if cfg.InjectEvery > 0 && step%cfg.InjectEvery == 0 {
			q := make([]float64, 256)
			k := make([]float64, 256)
			for i := range q {
				q[i] = float64(i+step%3) * 0.001
			}
			for i := range k {
				k[i] = float64((i*7+step)%5) * 0.001
			}
			gv := coupling.QKTCognitiveGravity(q, k, qwork, 16) * cfg.GravityKappa
			coupling.InjectGravityW(src, gv, 6, 6, 6, 0.15)
		}
		lattice4.StepHeat(cur, nxt, src, D, cfg.WaveDt)
		cur, nxt = nxt, cur
		hodge.RemoveMean(cur)

		if log != nil && (step == 0 || step == cfg.Steps-1 || step%(cfg.Steps/5+1) == 0) {
			log.Info("fourd step", "n", step, "center", cur.At(6, 6, 6, 4), "bias", coupling.LatticeToLogitBias(cur, 6, 6, 6, 4))
		}
		if w != nil && step == cfg.Steps-1 {
			_, _ = io.WriteString(w, "fourd_orchestrator_complete\n")
		}
		if cfg.Tick > 0 {
			time.Sleep(cfg.Tick)
		}
	}
	return nil
}
