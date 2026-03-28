// Package rq4dsidecar exposes a small inference hook for external binaries (e.g. 4dollama).
// It lives outside internal/ so dependents can import it while still using RQ4D quantum core.
package rq4dsidecar

import (
	"hash/fnv"
	"sync"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/quantum"
)

var (
	once   sync.Once
	sim    *quantum.Simulator
	simMu  sync.Mutex
)

func initSim() {
	once.Do(func() {
		sim = quantum.NewSimulator(4, 4, 4, 2, 1)
		sim.InitProductComputational()
		sim.SetUniformFields(0.07, 0.03)
		sim.JBond = 0.05
		sim.Dt = 0.04
	})
}

// SilentQuantumSteps advances the shared RQ4D lattice state from a prompt digest (no I/O).
func SilentQuantumSteps(prompt string) {
	initSim()
	h := fnv.New64a()
	_, _ = h.Write([]byte(prompt))
	n := int(h.Sum64()%4) + 1
	simMu.Lock()
	defer simMu.Unlock()
	for i := 0; i < n; i++ {
		sim.Step()
	}
}
