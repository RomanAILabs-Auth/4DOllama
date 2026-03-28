package rq4dcore

import (
	"encoding/hex"
	"net/http"
	_ "net/http/pprof" // registers on DefaultServeMux
	"sync/atomic"
)

// Profiler holds lightweight atomic counters for the runtime (no kernel hooks).
type Profiler struct {
	LatticeSteps      atomic.Uint64
	BondTasks         atomic.Uint64
	SchmidtTasks      atomic.Uint64
	Measurements      atomic.Uint64
	ExternalHandled   atomic.Uint64
	LogEvents         atomic.Uint64
	QueueDropsSim     atomic.Uint64
	QueueDropsExt     atomic.Uint64
	QueueDropsAI      atomic.Uint64
	AIEnqueued        atomic.Uint64
	AICompleted       atomic.Uint64
	ExecutorIdleNanos atomic.Uint64
}

// Snapshot returns a copy of counters for /metrics-style export.
func (p *Profiler) Snapshot() map[string]uint64 {
	if p == nil {
		return nil
	}
	return map[string]uint64{
		"lattice_steps":       p.LatticeSteps.Load(),
		"bond_tasks":          p.BondTasks.Load(),
		"schmidt_tasks":       p.SchmidtTasks.Load(),
		"measurements":        p.Measurements.Load(),
		"external_handled":    p.ExternalHandled.Load(),
		"log_events":          p.LogEvents.Load(),
		"queue_drops_sim":     p.QueueDropsSim.Load(),
		"queue_drops_ext":     p.QueueDropsExt.Load(),
		"queue_drops_ai":      p.QueueDropsAI.Load(),
		"ai_enqueued":         p.AIEnqueued.Load(),
		"ai_completed":        p.AICompleted.Load(),
		"executor_idle_nanos": p.ExecutorIdleNanos.Load(),
	}
}

// StartPprof binds runtime/pprof on addr (e.g. "127.0.0.1:6060"). User-space only.
func StartPprof(addr string) *http.Server {
	if addr == "" {
		return nil
	}
	s := &http.Server{Addr: addr}
	go func() { _ = s.ListenAndServe() }()
	return s
}

// StateHex returns SHA256 of state as hex (helper for JSON responses).
func StateHex(sum [32]byte) string {
	return hex.EncodeToString(sum[:])
}
