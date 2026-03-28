package rq4dcore

import (
	"encoding/hex"
	"sync"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/quantum"
)

// SharedEngine wraps the lattice simulator with an RWMutex so HTTP readers can
// snapshot hashes without racing the executor (zero-copy views: callers must
// not retain slices across lock release).
type SharedEngine struct {
	mu sync.RWMutex
	S  *quantum.Simulator
}

func NewSharedEngine(s *quantum.Simulator) *SharedEngine {
	return &SharedEngine{S: s}
}

// WithWrite runs fn with exclusive access to the simulator.
func (e *SharedEngine) WithWrite(fn func(*quantum.Simulator)) {
	e.mu.Lock()
	defer e.mu.Unlock()
	fn(e.S)
}

// WithRead runs fn with shared access (read-only use of S).
func (e *SharedEngine) WithRead(fn func(*quantum.Simulator)) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	fn(e.S)
}

// SnapshotDiagnostics returns norm², ⟨H⟩ mf, and state hash under read lock.
func (e *SharedEngine) SnapshotDiagnostics() (norm2, expectH float64, sha string) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.S == nil {
		return 0, 0, ""
	}
	norm2 = e.S.GlobalNorm2(e.S.ReA, e.S.ImA)
	expectH = e.S.ExpectationH()
	h := e.S.StateHashSHA256()
	return norm2, expectH, hex.EncodeToString(h[:])
}
