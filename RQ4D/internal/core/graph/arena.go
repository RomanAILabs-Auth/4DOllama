package graph

import (
	"sync"
)

// Arena is a bump allocator over a single backing slice (no per-op heap allocs).
type Arena struct {
	buf []byte
	off uint64
	mu  sync.Mutex // only for parallel graph build; hot execution path uses Reset under single writer
}

// NewArena preallocates nbytes of backing storage.
func NewArena(nbytes int) *Arena {
	if nbytes < 64 {
		nbytes = 64
	}
	return &Arena{buf: make([]byte, nbytes)}
}

// Cap returns total bytes capacity.
func (a *Arena) Cap() int {
	if a == nil {
		return 0
	}
	return len(a.buf)
}

// Used returns committed bytes.
func (a *Arena) Used() uint64 {
	if a == nil {
		return 0
	}
	return a.off
}

// Reset clears the bump pointer (logical free-all). O(1).
func (a *Arena) Reset() {
	if a == nil {
		return
	}
	a.mu.Lock()
	a.off = 0
	a.mu.Unlock()
}

// Alloc returns a zeroed slice of n bytes from the arena, or ErrArenaOOM.
func (a *Arena) Alloc(n int) ([]byte, error) {
	_, b, err := a.AllocTracked(n)
	return b, err
}

// AllocTracked returns the byte offset and slice for n bytes (8-byte aligned start).
func (a *Arena) AllocTracked(n int) (offset uint64, dst []byte, err error) {
	if n < 0 {
		return 0, nil, ErrArenaOOM
	}
	if n == 0 {
		a.mu.Lock()
		off := a.off
		a.mu.Unlock()
		return off, nil, nil
	}
	align := (8 - int(a.off&7)) & 7
	a.mu.Lock()
	defer a.mu.Unlock()
	need := uint64(align + n)
	if a.off+need > uint64(len(a.buf)) {
		return 0, nil, ErrArenaOOM
	}
	a.off += uint64(align)
	start := a.off
	a.off += uint64(n)
	s := a.buf[start:a.off]
	clear(s)
	return start, s, nil
}

// View returns a sub-slice at offset/len without allocating (must be in bounds).
func (a *Arena) View(offset uint64, length uint32) []byte {
	if a == nil || length == 0 {
		return nil
	}
	end := offset + uint64(length)
	if end > uint64(len(a.buf)) {
		return nil
	}
	return a.buf[offset:end]
}

// NodePool is a fixed-capacity dense store for Node structs (no map churn in hot paths).
type NodePool struct {
	nodes []Node
	ids   []NodeID
	next  NodeID
}

// NewNodePool preallocates max nodes.
func NewNodePool(max int) *NodePool {
	if max < 4 {
		max = 4
	}
	return &NodePool{
		nodes: make([]Node, 0, max),
		ids:   make([]NodeID, 0, max),
		next:  1,
	}
}

// Len active nodes.
func (p *NodePool) Len() int { return len(p.nodes) }

// Cap max nodes.
func (p *NodePool) Cap() int { return cap(p.nodes) }

// Append adds a node and returns its dense index and ID.
func (p *NodePool) Append(n Node) (idx int, id NodeID, err error) {
	if len(p.nodes) >= cap(p.nodes) {
		return 0, 0, ErrTooManyNodes
	}
	id = p.next
	p.next++
	n.ID = id
	idx = len(p.nodes)
	n.TopoIdx = idx
	p.nodes = append(p.nodes, n)
	p.ids = append(p.ids, id)
	return idx, id, nil
}

// Get returns node by dense index.
func (p *NodePool) Get(idx int) *Node {
	if idx < 0 || idx >= len(p.nodes) {
		return nil
	}
	return &p.nodes[idx]
}

// IdxOfID returns dense index for NodeID, or -1.
func (p *NodePool) IdxOfID(id NodeID) int {
	for i := range p.nodes {
		if p.nodes[i].ID == id {
			return i
		}
	}
	return -1
}

// Nodes returns the backing slice (read-only contract: do not append).
func (p *NodePool) Nodes() []Node { return p.nodes }

// EdgePool holds edges with stable indices.
type EdgePool struct {
	edges []Edge
}

func NewEdgePool(max int) *EdgePool {
	if max < 4 {
		max = 4
	}
	return &EdgePool{edges: make([]Edge, 0, max)}
}

func (p *EdgePool) Append(e Edge) (int, error) {
	if len(p.edges) >= cap(p.edges) {
		return 0, ErrTooManyEdges
	}
	i := len(p.edges)
	p.edges = append(p.edges, e)
	return i, nil
}

func (p *EdgePool) Edges() []Edge { return p.edges }