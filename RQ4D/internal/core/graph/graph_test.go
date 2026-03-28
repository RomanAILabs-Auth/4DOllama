package graph

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestMacroDAGTopo(t *testing.T) {
	cfg := DefaultGraphConfig()
	cfg.MaxNodes = 128
	cfg.MaxEdges = 256
	cfg.ArenaBytes = 4096
	g := NewGraph(cfg)
	meta := CostMeta{ElementCount: 50_000, FLOPEstimate: 100}
	p := make([]byte, 16)
	binary.LittleEndian.PutUint64(p[0:], math.Float64bits(2))
	binary.LittleEndian.PutUint64(p[8:], math.Float64bits(3))
	a, err := g.AddMacroNode(OpScaleF64, meta, p)
	if err != nil {
		t.Fatal(err)
	}
	meta2 := CostMeta{ElementCount: 50_000}
	b, err := g.AddMacroNode(OpAddF64, meta2, p)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(a, b, 0.5); err != nil {
		t.Fatal(err)
	}
	if err := g.Finalize(); err != nil {
		t.Fatal(err)
	}
	layers, err := TopologicalLayers(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != 2 {
		t.Fatalf("layers: %+v", layers)
	}
	var buf bytes.Buffer
	n, err := g.WriteTo(&buf)
	if err != nil || n < 64 {
		t.Fatalf("write %d %v", n, err)
	}
}

func TestCycleDetected(t *testing.T) {
	cfg := DefaultGraphConfig()
	cfg.MaxNodes = 16
	cfg.MaxEdges = 32
	cfg.ArenaBytes = 1024
	g := NewGraph(cfg)
	meta := CostMeta{ElementCount: 10_000}
	a, _ := g.AddMacroNode(OpNoOp, meta, nil)
	b, _ := g.AddMacroNode(OpNoOp, meta, nil)
	c, _ := g.AddMacroNode(OpNoOp, meta, nil)
	_ = g.AddEdge(a, b, 1)
	_ = g.AddEdge(b, c, 1)
	_ = g.AddEdge(c, a, 1)
	if err := g.Finalize(); err != nil {
		t.Fatal(err)
	}
	if _, err := TopologicalLayers(g); err != ErrDAGCycle {
		t.Fatalf("want cycle got %v", err)
	}
}
