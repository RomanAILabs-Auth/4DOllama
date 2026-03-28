package graph

import (
	"encoding/binary"
	"math"
)

// RunMacroOp executes a pure macro-kernel over payload bytes (SIMD-friendly scalar loop).
// Hot path: no heap allocation when payload is fixed.
func RunMacroOp(op OpType, elementCount int64, payload []byte, weight float64) {
	if len(payload) == 0 {
		return
	}
	switch op {
	case OpNoOp, OpLatticeLocal, OpBondMix:
		return
	case OpScaleF64:
		scaleF64Payload(payload, weight)
	case OpAddF64:
		addConstF64Payload(payload, weight)
	case OpCopy:
		// no-op if single buffer; used for future two-buffer ops
		_ = elementCount
	}
}

func scaleF64Payload(b []byte, s float64) {
	n := len(b) / 8
	for i := 0; i < n; i++ {
		o := i * 8
		v := math.Float64frombits(binary.LittleEndian.Uint64(b[o:]))
		v *= s
		binary.LittleEndian.PutUint64(b[o:], math.Float64bits(v))
	}
}

func addConstF64Payload(b []byte, c float64) {
	n := len(b) / 8
	for i := 0; i < n; i++ {
		o := i * 8
		v := math.Float64frombits(binary.LittleEndian.Uint64(b[o:]))
		v += c
		binary.LittleEndian.PutUint64(b[o:], math.Float64bits(v))
	}
}
