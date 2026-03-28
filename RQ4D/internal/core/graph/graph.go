package graph

import (
	"errors"
	"slices"
)

// Graph is a DAG of macro-nodes with arena-backed payloads and adjacency for traversal.
type Graph struct {
	cfg GraphConfig

	pool   *NodePool
	edges  *EdgePool
	arena  *Arena
	id2idx map[NodeID]int
	adjOut [][]int // edge indices leaving node idx
	adjIn  [][]int // edge indices entering node idx
	final  bool
}

// NewGraph constructs an empty graph with preallocated pools.
func NewGraph(cfg GraphConfig) *Graph {
	if cfg.MaxNodes <= 0 {
		cfg.MaxNodes = 1 << 16
	}
	if cfg.MaxEdges <= 0 {
		cfg.MaxEdges = 1 << 18
	}
	if cfg.ArenaBytes <= 0 {
		cfg.ArenaBytes = 1 << 24
	}
	return &Graph{
		cfg:    cfg,
		pool:   NewNodePool(cfg.MaxNodes),
		edges:  NewEdgePool(cfg.MaxEdges),
		arena:  NewArena(cfg.ArenaBytes),
		id2idx: make(map[NodeID]int, minInt(1024, cfg.MaxNodes)),
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Arena returns the backing bump allocator (shared by all nodes).
func (g *Graph) Arena() *Arena { return g.arena }

// Config returns a copy of graph limits.
func (g *Graph) Config() GraphConfig { return g.cfg }

// NodeCount returns the number of macro-nodes.
func (g *Graph) NodeCount() int { return g.pool.Len() }

// EdgeCount returns dependency edges.
func (g *Graph) EdgeCount() int { return len(g.edges.Edges()) }

// Nodes exposes the dense node slice (do not append; indices are stable until mutation).
func (g *Graph) Nodes() []Node { return g.pool.Nodes() }

// NodeByID returns a pointer into the pool or nil.
func (g *Graph) NodeByID(id NodeID) *Node {
	idx, ok := g.id2idx[id]
	if !ok {
		return nil
	}
	return g.pool.Get(idx)
}

// IdxOfID maps NodeID to dense index.
func (g *Graph) IdxOfID(id NodeID) (int, bool) {
	idx, ok := g.id2idx[id]
	return idx, ok
}

// AddMacroNode registers a macro-chunk vertex. payload is copied into the arena once.
func (g *Graph) AddMacroNode(op OpType, meta CostMeta, payload []byte) (NodeID, error) {
	if g.final {
		return 0, errors.New("graph: finalized")
	}
	if g.cfg.StrictMacroNodes && meta.ElementCount < RecommendedMacroMinElements {
		return 0, ErrInvalidMacroNode
	}
	if meta.ElementCount < HardMacroMinElements {
		return 0, ErrInvalidMacroNode
	}
	var off uint64
	var plen uint32
	if len(payload) > 0 {
		o, b, err := g.arena.AllocTracked(len(payload))
		if err != nil {
			return 0, err
		}
		copy(b, payload)
		off = o
		plen = uint32(len(payload))
	}
	meta.MemoryFootprintBytes = plen
	n := Node{
		Op:            op,
		Meta:          meta,
		PayloadOffset: off,
		PayloadLen:    plen,
	}
	idx, id, err := g.pool.Append(n)
	if err != nil {
		return 0, err
	}
	g.id2idx[id] = idx
	return id, nil
}

// AddEdge adds a dependency From -> To (From must finish before To).
func (g *Graph) AddEdge(from, to NodeID, weight float64) error {
	if g.final {
		return errors.New("graph: finalized")
	}
	fi, ok := g.id2idx[from]
	if !ok {
		return ErrUnknownNode
	}
	ti, ok := g.id2idx[to]
	if !ok {
		return ErrUnknownNode
	}
	if fi == ti {
		return ErrDAGCycle
	}
	_, err := g.edges.Append(Edge{From: from, To: to, Weight: weight})
	if err != nil {
		return err
	}
	// adjacency filled in Finalize for cache-friendly packed slices
	_ = fi
	_ = ti
	return nil
}

// Finalize builds adjacency lists and marks the graph immutable for structure changes.
func (g *Graph) Finalize() error {
	if g.final {
		return nil
	}
	n := g.pool.Len()
	g.adjOut = make([][]int, n)
	g.adjIn = make([][]int, n)
	for i := range g.adjOut {
		g.adjOut[i] = make([]int, 0, 4)
		g.adjIn[i] = make([]int, 0, 4)
	}
	for ei, e := range g.edges.Edges() {
		ui, ok := g.id2idx[e.From]
		if !ok {
			return ErrUnknownNode
		}
		vi, ok := g.id2idx[e.To]
		if !ok {
			return ErrUnknownNode
		}
		g.adjOut[ui] = append(g.adjOut[ui], ei)
		g.adjIn[vi] = append(g.adjIn[vi], ei)
	}
	for i := 0; i < g.pool.Len(); i++ {
		if p := g.pool.Get(i); p != nil {
			p.TopoIdx = i
		}
	}
	g.final = true
	return nil
}

// AdjOut returns outgoing edge indices for dense node index.
func (g *Graph) AdjOut(nodeIdx int) []int {
	if nodeIdx < 0 || nodeIdx >= len(g.adjOut) {
		return nil
	}
	return g.adjOut[nodeIdx]
}

// AdjIn returns incoming edge indices for dense node index.
func (g *Graph) AdjIn(nodeIdx int) []int {
	if nodeIdx < 0 || nodeIdx >= len(g.adjIn) {
		return nil
	}
	return g.adjIn[nodeIdx]
}

// Edges returns the edge slice (read-only).
func (g *Graph) Edges() []Edge { return g.edges.Edges() }

// ValidateDAG checks for cycles via Kahn (requires Finalize).
func (g *Graph) ValidateDAG() error {
	if !g.final {
		if err := g.Finalize(); err != nil {
			return err
		}
	}
	_, err := TopologicalLayers(g)
	return err
}

// PayloadView returns arena bytes for node n (zero-copy).
func (g *Graph) PayloadView(n *Node) []byte {
	if n == nil || g.arena == nil {
		return nil
	}
	return g.arena.View(n.PayloadOffset, n.PayloadLen)
}

// CloneIDs returns a deterministic sorted copy of all NodeIDs (serialization helper).
func (g *Graph) CloneIDs() []NodeID {
	out := make([]NodeID, 0, g.pool.Len())
	for _, n := range g.pool.Nodes() {
		out = append(out, n.ID)
	}
	slices.Sort(out)
	return out
}

// LayerNodePtrs fills buf with pointers to nodes in dense index order for this layer.
// Reuses buf capacity to avoid per-wave heap allocation when cap(buf) >= len(indices).
// Pointers reference the graph's stable node storage (valid after Finalize, no further structural changes).
func (g *Graph) LayerNodePtrs(indices []int, buf []*Node) []*Node {
	nodes := g.pool.Nodes()
	if cap(buf) < len(indices) {
		buf = make([]*Node, len(indices))
	} else {
		buf = buf[:len(indices)]
	}
	for i, idx := range indices {
		if idx < 0 || idx >= len(nodes) {
			buf[i] = nil
			continue
		}
		buf[i] = &nodes[idx]
	}
	return buf
}
