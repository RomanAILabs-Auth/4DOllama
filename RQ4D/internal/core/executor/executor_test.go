package executor

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/core/graph"
)

func TestCPUBackendWavefront(t *testing.T) {
	cfg := graph.DefaultGraphConfig()
	cfg.MaxNodes = 64
	cfg.MaxEdges = 128
	cfg.ArenaBytes = 8192
	g := graph.NewGraph(cfg)
	meta := graph.CostMeta{ElementCount: 12_000}
	p := make([]byte, 8)
	binary.LittleEndian.PutUint64(p, math.Float64bits(4))
	a, _ := g.AddMacroNode(graph.OpScaleF64, meta, p)
	b, _ := g.AddMacroNode(graph.OpScaleF64, meta, p)
	_ = g.AddEdge(a, b, 1)
	_ = g.Finalize()

	tel := &Telemetry{}
	be := NewCPUBackend(g, Options{Workers: 2, Telemetry: tel})
	buf := make([]*graph.Node, 0, 8)
	if err := RunGraphWavefronts(g, be, tel, buf); err != nil {
		t.Fatal(err)
	}
	if tel.NodesExecuted.Load() != 2 {
		t.Fatalf("nodes executed %d", tel.NodesExecuted.Load())
	}
}

func TestNewBackendGPUMock(t *testing.T) {
	cfg := graph.DefaultGraphConfig()
	cfg.MaxNodes = 8
	cfg.MaxEdges = 16
	cfg.ArenaBytes = 1024
	g := graph.NewGraph(cfg)
	meta := graph.CostMeta{ElementCount: 10_000}
	_, _ = g.AddMacroNode(graph.OpNoOp, meta, nil)
	_ = g.Finalize()
	b, err := NewBackend(BackendGPU, g, Options{Workers: 2, Telemetry: &Telemetry{}})
	if err != nil {
		t.Fatal(err)
	}
	_ = b.Shutdown()
}
