package executor

import (
	"runtime"
	"sync"
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/core/graph"
)

// CPUBackend runs macro-node batches on a fixed worker pool (minimal scheduling overhead).
type CPUBackend struct {
	g       *graph.Graph
	workers int
	tel     *Telemetry
}

// NewCPUBackend constructs a CPU executor. Graph must be finalized.
func NewCPUBackend(g *graph.Graph, opts Options) *CPUBackend {
	w := opts.Workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	if w < 1 {
		w = 1
	}
	return &CPUBackend{g: g, workers: w, tel: opts.Telemetry}
}

func (c *CPUBackend) Initialize() error { return nil }

func (c *CPUBackend) Sync() error { return nil }

func (c *CPUBackend) Shutdown() error { return nil }

// ExecuteBatch runs independent nodes in parallel chunks (same wavefront).
func (c *CPUBackend) ExecuteBatch(nodes []*graph.Node) error {
	if len(nodes) == 0 {
		return nil
	}
	n := len(nodes)
	w := c.workers
	if w > n {
		w = n
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	wg.Add(w)
	for t := 0; t < w; t++ {
		lo := t * chunk
		if lo >= n {
			wg.Done()
			continue
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				c.execOne(nodes[i])
			}
		}(lo, hi)
	}
	wg.Wait()
	if c.tel != nil {
		c.tel.ObserveBatch()
		c.tel.BumpBackendUtil(uint64(n))
	}
	return nil
}

func (c *CPUBackend) execOne(node *graph.Node) {
	if node == nil {
		return
	}
	tEdge := time.Now()
	w := nodeEdgeWeight(c.g, node)
	if c.tel != nil {
		c.tel.ObserveEdgeWalk(time.Since(tEdge))
	}
	t0 := time.Now()
	payload := c.g.PayloadView(node)
	graph.RunMacroOp(node.Op, node.Meta.ElementCount, payload, w)
	if c.tel != nil {
		c.tel.ObserveNode(time.Since(t0))
	}
}

// nodeEdgeWeight uses mean inbound edge weight as coupling factor (O(indegree)); telemetry-friendly.
func nodeEdgeWeight(g *graph.Graph, node *graph.Node) float64 {
	if g == nil || node == nil {
		return 1
	}
	idx := node.TopoIdx
	if idx < 0 {
		return 1
	}
	ins := g.AdjIn(idx)
	if len(ins) == 0 {
		return 1
	}
	edges := g.Edges()
	var s float64
	for _, ei := range ins {
		s += edges[ei].Weight
	}
	return s / float64(len(ins))
}
