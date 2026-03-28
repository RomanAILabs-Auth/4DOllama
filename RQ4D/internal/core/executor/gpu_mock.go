package executor

import (
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/core/graph"
)

// GPUMockBackend is a pure-Go "device" that reuses the CPU worker pool for kernel execution
// and adds a tiny Sync delay to mimic queue fencing (no CGO, no CUDA).
type GPUMockBackend struct {
	inner *CPUBackend
}

// NewGPUMockBackend creates a mock GPU executor.
func NewGPUMockBackend(g *graph.Graph, opts Options) *GPUMockBackend {
	streams := opts.Workers
	if streams <= 0 {
		streams = 4
	}
	if streams > 64 {
		streams = 64
	}
	opts.Workers = streams
	return &GPUMockBackend{inner: NewCPUBackend(g, opts)}
}

func (m *GPUMockBackend) Initialize() error { return m.inner.Initialize() }

func (m *GPUMockBackend) ExecuteBatch(nodes []*graph.Node) error {
	return m.inner.ExecuteBatch(nodes)
}

func (m *GPUMockBackend) Sync() error {
	time.Sleep(time.Microsecond)
	return m.inner.Sync()
}

func (m *GPUMockBackend) Shutdown() error { return m.inner.Shutdown() }
