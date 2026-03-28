package rq4dcore

import (
	"sync/atomic"
)

// TaskKind identifies scheduler work units (compute graph nodes).
type TaskKind uint8

const (
	TaskLatticeStep TaskKind = iota
	TaskBondUpdate
	TaskSchmidtTruncate
	TaskMeasurementSample
	TaskExternalRequest
	TaskAILoadModel
	TaskAISaveModel
	TaskAILayerClifford
	TaskAIRotate4D
	TaskAITrainStep
	TaskAIInference
	TaskAIEvolveField
)

// Task is a single unit of work for the executor goroutine.
// External callers may set Resp for synchronous completion (buffered cap ≥ 1).
type Task struct {
	Kind        TaskKind
	Site        int
	Steps       int
	Seed        int64 // measurement RNG; 0 → default
	WarmupSteps int   // for TaskExternalRequest: optional evolution before snapshot
	Prompt      string
	Resp        chan TaskResult
	// AI tier (romanai.4dai / Roma4D control); AIBody holds JSON float array for INFER when set.
	AIPath   string
	AILR     float64
	AIPlane  string
	AIAngle  float64
	AIField  string
	AIBody   string
	AICliffN int // LAYER CLIFFORD size=N
}

// TaskResult carries optional synchronous reply metadata.
type TaskResult struct {
	Err       error
	StateSHA  string
	Norm2     float64
	ExpectH   float64
	MeasureK  int
	Transform string
	AIOut     []float64 // inference output (copy for consumer)
	AIMessage string    // human-readable side channel
}

// spscRing is a single-producer single-consumer bounded queue using atomics.
// Only the coordinator goroutine may Push; only the executor goroutine may Pop.
type spscRing struct {
	mask   uint64
	slots  []Task
	_      [56]byte // reduce false sharing
	head   atomic.Uint64
	tail   atomic.Uint64
}

func newSPSCRing(capPow2 int) *spscRing {
	if capPow2 < 16 {
		capPow2 = 16
	}
	if capPow2&(capPow2-1) != 0 {
		panic("rq4dcore: ring cap must be power of two")
	}
	return &spscRing{
		mask:  uint64(capPow2) - 1,
		slots: make([]Task, capPow2),
	}
}

// push enqueues one task. Returns false if full.
func (r *spscRing) push(t Task) bool {
	tail := r.tail.Load()
	next := tail + 1
	if next-r.head.Load() > r.mask+1 {
		return false
	}
	r.slots[tail&r.mask] = t
	r.tail.Store(next)
	return true
}

// pop dequeues one task. Returns false if empty.
func (r *spscRing) pop() (Task, bool) {
	head := r.head.Load()
	if head == r.tail.Load() {
		return Task{}, false
	}
	t := r.slots[head&r.mask]
	r.head.Store(head + 1)
	return t, true
}
