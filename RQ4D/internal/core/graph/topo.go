package graph

import "slices"

// TopologicalLayers returns execution wavefronts: layer[k] are dense node indices
// with all inbound dependencies satisfied by prior layers. Deterministic tie-break by ascending index.
func TopologicalLayers(g *Graph) ([][]int, error) {
	if !g.final {
		if err := g.Finalize(); err != nil {
			return nil, err
		}
	}
	n := g.pool.Len()
	indeg := make([]int, n)
	edges := g.edges.Edges()
	for _, e := range edges {
		u, ok := g.id2idx[e.From]
		if !ok {
			return nil, ErrUnknownNode
		}
		v, ok := g.id2idx[e.To]
		if !ok {
			return nil, ErrUnknownNode
		}
		_ = u
		indeg[v]++
	}
	var layers [][]int
	var cur []int
	for i := 0; i < n; i++ {
		if indeg[i] == 0 {
			cur = append(cur, i)
		}
	}
	slices.Sort(cur)
	for len(cur) > 0 {
		layers = append(layers, cur)
		next := make([]int, 0, n)
		for _, u := range cur {
			for _, ei := range g.adjOut[u] {
				e := edges[ei]
				v := g.id2idx[e.To]
				indeg[v]--
				if indeg[v] == 0 {
					next = append(next, v)
				}
			}
		}
		slices.Sort(next)
		cur = next
	}
	if len(layers) == 0 && n > 0 {
		return nil, ErrDAGCycle
	}
	processed := 0
	for _, L := range layers {
		processed += len(L)
	}
	if processed != n {
		return nil, ErrDAGCycle
	}
	return layers, nil
}
