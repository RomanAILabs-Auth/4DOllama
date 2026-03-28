// Package graph provides a deterministic DAG model for macro-chunk tensor/state execution.
// MACRO-NODE RULE: each Node must cover a large batch of manifold/tensor elements (see
// RecommendedMacroMinElements). Never map one logical element to one DAG node at scale.
package graph

import (
	"errors"
	"fmt"
)

// RecommendedMacroMinElements is the soft minimum elements per macro-node for scheduler health.
const RecommendedMacroMinElements int64 = 10_000

// HardMacroMinElements is enforced by AddMacroNode when StrictMacroNodes is true.
const HardMacroMinElements int64 = 1

// OpType identifies a pure compute kernel (no hidden state inside the op).
type OpType uint16

const (
	OpNoOp OpType = iota
	OpScaleF64
	OpAddF64
	OpCopy
	OpLatticeLocal // placeholder for future fused lattice local pass
	OpBondMix      // placeholder for fused bond / coupling chunk
)

// String returns a stable name for logs and serialization.
func (o OpType) String() string {
	switch o {
	case OpNoOp:
		return "nop"
	case OpScaleF64:
		return "scale_f64"
	case OpAddF64:
		return "add_f64"
	case OpCopy:
		return "copy"
	case OpLatticeLocal:
		return "lattice_local"
	case OpBondMix:
		return "bond_mix"
	default:
		return fmt.Sprintf("op_%d", o)
	}
}

// ParseOpType reverses String for deterministic deserialization.
func ParseOpType(s string) (OpType, error) {
	switch s {
	case "nop":
		return OpNoOp, nil
	case "scale_f64":
		return OpScaleF64, nil
	case "add_f64":
		return OpAddF64, nil
	case "copy":
		return OpCopy, nil
	case "lattice_local":
		return OpLatticeLocal, nil
	case "bond_mix":
		return OpBondMix, nil
	default:
		return 0, fmt.Errorf("graph: unknown op %q", s)
	}
}

// NodeID is an opaque stable identifier (monotonic, never reused within a Graph build session).
type NodeID uint64

// CostMeta carries abstract scheduling hints (not wall-clock truth until measured).
type CostMeta struct {
	// ElementCount is how many scalar/tensor elements this macro-node logically covers.
	ElementCount int64
	// FLOPEstimate is an optional abstract cost unit (e.g. scaled FLOPs).
	FLOPEstimate uint64
	// MemoryFootprintBytes is optional footprint of Payload in the arena.
	MemoryFootprintBytes uint32
}

// Node is a macro-chunk compute vertex. All inputs/outputs are explicit edges.
type Node struct {
	ID   NodeID
	Op   OpType
	Meta CostMeta

	// PayloadOffset/PayloadLen address a sub-slice of the Graph arena (zero-copy view).
	PayloadOffset uint64
	PayloadLen    uint32

	// TopoIdx is the dense execution index [0,n) after Graph.Finalize.
	TopoIdx int
}

// Edge is a typed dependency: From must complete before To.
type Edge struct {
	From   NodeID
	To     NodeID
	Weight float64
}

// GraphConfig caps storage for arena-style preallocation (zero-GC steady state).
type GraphConfig struct {
	MaxNodes           int
	MaxEdges           int
	ArenaBytes         int
	StrictMacroNodes   bool // if true, ElementCount < RecommendedMacroMinElements is an error
	AllowDenseWarnings bool // if false and StrictMacroNodes false, still warn via callback optional
}

// DefaultGraphConfig returns conservative production-ish defaults.
func DefaultGraphConfig() GraphConfig {
	return GraphConfig{
		MaxNodes:           1 << 20,
		MaxEdges:           1 << 22,
		ArenaBytes:         1 << 30, // 1 GiB cap; caller can shrink
		StrictMacroNodes:   false,
		AllowDenseWarnings: true,
	}
}

var (
	ErrTooManyNodes     = errors.New("graph: max nodes exceeded")
	ErrTooManyEdges     = errors.New("graph: max edges exceeded")
	ErrArenaOOM         = errors.New("graph: arena out of memory")
	ErrDAGCycle         = errors.New("graph: cycle detected")
	ErrUnknownNode      = errors.New("graph: unknown node id")
	ErrInvalidMacroNode = errors.New("graph: macro-node element count below policy")
)
