package graph

import (
	"encoding/binary"
	"io"
	"slices"
)

const serialMagic uint32 = 0x52313444 // 'R' '1' '4' 'D'

// WriteTo writes a deterministic binary snapshot: magic, version, nodes, edges (sorted by From,To).
// Payload bytes are not inlined (only offsets/lengths); pair with separate arena dump if needed.
func (g *Graph) WriteTo(w io.Writer) (int64, error) {
	if !g.final {
		if err := g.Finalize(); err != nil {
			return 0, err
		}
	}
	var n int64
	hdr := make([]byte, 8)
	binary.LittleEndian.PutUint32(hdr[0:4], serialMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], 1) // format version
	nn, err := w.Write(hdr)
	n += int64(nn)
	if err != nil {
		return n, err
	}
	nodes := g.pool.Nodes()
	binary.LittleEndian.PutUint64(hdr[0:8], uint64(len(nodes)))
	nn, err = w.Write(hdr[:8])
	n += int64(nn)
	if err != nil {
		return n, err
	}
	for i := range nodes {
		rec := marshalNode(&nodes[i])
		nn, err = w.Write(rec)
		n += int64(nn)
		if err != nil {
			return n, err
		}
	}
	edges := g.edges.Edges()
	type pair struct {
		e Edge
	}
	ps := make([]pair, len(edges))
	for i := range edges {
		ps[i] = pair{e: edges[i]}
	}
	slices.SortFunc(ps, func(a, b pair) int {
		if a.e.From != b.e.From {
			return int(a.e.From - b.e.From)
		}
		return int(a.e.To - b.e.To)
	})
	binary.LittleEndian.PutUint64(hdr[0:8], uint64(len(ps)))
	nn, err = w.Write(hdr[:8])
	n += int64(nn)
	if err != nil {
		return n, err
	}
	for _, p := range ps {
		rec := marshalEdge(&p.e)
		nn, err = w.Write(rec)
		n += int64(nn)
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func marshalNode(node *Node) []byte {
	b := make([]byte, 56)
	binary.LittleEndian.PutUint64(b[0:8], uint64(node.ID))
	binary.LittleEndian.PutUint16(b[8:10], uint16(node.Op))
	binary.LittleEndian.PutUint64(b[16:24], uint64(node.Meta.ElementCount))
	binary.LittleEndian.PutUint64(b[24:32], node.Meta.FLOPEstimate)
	binary.LittleEndian.PutUint32(b[32:36], node.Meta.MemoryFootprintBytes)
	binary.LittleEndian.PutUint64(b[36:44], node.PayloadOffset)
	binary.LittleEndian.PutUint32(b[44:48], node.PayloadLen)
	binary.LittleEndian.PutUint32(b[48:52], uint32(node.TopoIdx))
	return b
}

func marshalEdge(e *Edge) []byte {
	b := make([]byte, 32)
	binary.LittleEndian.PutUint64(b[0:8], uint64(e.From))
	binary.LittleEndian.PutUint64(b[8:16], uint64(e.To))
	binary.LittleEndian.PutUint64(b[16:24], uint64(e.Weight))
	return b
}
