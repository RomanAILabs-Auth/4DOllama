package executor

import (
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/core/graph"
)

// RunGraphWavefronts performs topological wavefront execution using backend.ExecuteBatch per layer.
// batchBuf is reused across layers to avoid per-layer []* allocation when capacity suffices.
func RunGraphWavefronts(g *graph.Graph, b ExecutorBackend, tel *Telemetry, batchBuf []*graph.Node) error {
	if err := b.Initialize(); err != nil {
		return err
	}
	defer func() { _ = b.Shutdown() }()

	layers, err := graph.TopologicalLayers(g)
	if err != nil {
		return err
	}
	var ptrBuf []*graph.Node
	if cap(batchBuf) > 0 {
		ptrBuf = batchBuf[:0]
	}
	var lastBatchEnd time.Time
	for li, layer := range layers {
		if li > 0 && tel != nil && !lastBatchEnd.IsZero() {
			tel.ObserveSchedulerIdle(time.Since(lastBatchEnd))
		}
		ptrBuf = g.LayerNodePtrs(layer, ptrBuf)
		if err := b.ExecuteBatch(ptrBuf); err != nil {
			return err
		}
		if err := b.Sync(); err != nil {
			return err
		}
		lastBatchEnd = time.Now()
	}
	return nil
}
