// Package executor provides backend abstraction for macro-DAG execution (CPU + pure-Go GPU mock).
package executor

import (
	"errors"
	"fmt"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/core/graph"
)

// ExecutorBackend runs batches of macro-nodes (already dependency-ordered per wavefront).
// Implementations must not retain Node pointers across Sync boundaries unless documented.
type ExecutorBackend interface {
	Initialize() error
	ExecuteBatch(nodes []*graph.Node) error
	Sync() error
	Shutdown() error
}

// BackendKind selects runtime backend (pure Go builds only).
type BackendKind string

const (
	BackendCPU  BackendKind = "cpu"
	BackendGPU  BackendKind = "gpu"
	BackendAuto BackendKind = "auto"
)

// ParseBackendKind normalizes CLI flags.
func ParseBackendKind(s string) (BackendKind, error) {
	switch s {
	case "cpu", "CPU":
		return BackendCPU, nil
	case "gpu", "GPU":
		return BackendGPU, nil
	case "auto", "AUTO":
		return BackendAuto, nil
	default:
		return "", fmt.Errorf("executor: unknown backend %q", s)
	}
}

// NewBackend constructs the requested backend. g must be finalized.
func NewBackend(kind BackendKind, g *graph.Graph, opts Options) (ExecutorBackend, error) {
	if g == nil {
		return nil, errors.New("executor: nil graph")
	}
	if err := g.ValidateDAG(); err != nil {
		return nil, err
	}
	switch kind {
	case BackendCPU:
		return NewCPUBackend(g, opts), nil
	case BackendGPU:
		return NewGPUMockBackend(g, opts), nil
	case BackendAuto:
		// Pure-Go phase: auto prefers CPU; GPU mock available explicitly.
		return NewCPUBackend(g, opts), nil
	default:
		return nil, fmt.Errorf("executor: unsupported kind %v", kind)
	}
}

// Options tunes worker pool and telemetry.
type Options struct {
	Workers int // 0 → GOMAXPROCS(0)
	Telemetry *Telemetry
}
