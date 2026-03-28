// Package rq4dbridge runs RQ4D quantum lattice evolution and the local Cl(4,0) inference lattice
// on every completion request. Output text is unchanged; this work stays off the user-visible path.
package rq4dbridge

import (
	"github.com/RomanAILabs-Auth/RomaQuantum4D/rq4dsidecar"

	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/inference"
)

const promptCap = 256

func promptFloats(prompt string) []float32 {
	if prompt == "" {
		return nil
	}
	out := make([]float32, 0, min(len(prompt), promptCap))
	for _, r := range prompt {
		if len(out) >= promptCap {
			break
		}
		out = append(out, float32(r))
	}
	return out
}

// SilentInferencePass advances RQ4D quantum state and one geometric lattice tick derived from the prompt.
// Safe to call on every HTTP generate/chat; failures are ignored.
func SilentInferencePass(eng engine.Engine, prompt string) {
	rq4dsidecar.SilentQuantumSteps(prompt)

	if eng == nil {
		return
	}
	fs := promptFloats(prompt)
	if len(fs) == 0 {
		return
	}
	rope, err := eng.Rope4DSequence(fs)
	if err != nil || len(rope) < 4 {
		return
	}
	sl := len(rope) / 4
	attn, aerr := eng.SpacetimeAttention4D(rope, rope, rope, sl)
	if aerr != nil || len(attn) < 4 {
		attn = rope
	}
	_ = inference.GlobalLattice().OnTokenStepRuntime4D(rope, attn, nil, 0)
}
