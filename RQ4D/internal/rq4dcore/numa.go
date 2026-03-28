package rq4dcore

import "runtime"

// WorkerProcs returns a conservative worker count (user-space). True NUMA pinning
// would require platform-specific APIs; we expose a single knob via runtime.
func WorkerProcs() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		return 1
	}
	return n
}
