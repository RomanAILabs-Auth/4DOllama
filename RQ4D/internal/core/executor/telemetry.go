package executor

import (
	"fmt"
	"io"
	"sync/atomic"
	"time"
)

// Telemetry holds always-on atomic metrics (lock-free reads for /metrics scraping).
type Telemetry struct {
	NodesExecuted    atomic.Uint64
	NodeExecNanos    atomic.Uint64
	BatchesExecuted  atomic.Uint64
	SchedulerIdleNs  atomic.Uint64
	EdgeWalkNanos    atomic.Uint64
	BackendUtilTicks atomic.Uint64 // abstract utilization counter
}

// WritePrometheusText emits a minimal text exposition (no registry dependency).
func (t *Telemetry) WritePrometheusText(w io.Writer) {
	if t == nil {
		return
	}
	fmt.Fprintf(w, "# TYPE rq4d_graph_nodes_executed counter\n")
	fmt.Fprintf(w, "rq4d_graph_nodes_executed %d\n", t.NodesExecuted.Load())
	fmt.Fprintf(w, "# TYPE rq4d_graph_node_exec_nanoseconds counter\n")
	fmt.Fprintf(w, "rq4d_graph_node_exec_nanoseconds %d\n", t.NodeExecNanos.Load())
	fmt.Fprintf(w, "# TYPE rq4d_graph_batches_executed counter\n")
	fmt.Fprintf(w, "rq4d_graph_batches_executed %d\n", t.BatchesExecuted.Load())
	fmt.Fprintf(w, "# TYPE rq4d_graph_scheduler_idle_nanoseconds counter\n")
	fmt.Fprintf(w, "rq4d_graph_scheduler_idle_nanoseconds %d\n", t.SchedulerIdleNs.Load())
	fmt.Fprintf(w, "# TYPE rq4d_graph_edge_walk_nanoseconds counter\n")
	fmt.Fprintf(w, "rq4d_graph_edge_walk_nanoseconds %d\n", t.EdgeWalkNanos.Load())
	fmt.Fprintf(w, "# TYPE rq4d_graph_backend_util_ticks counter\n")
	fmt.Fprintf(w, "rq4d_graph_backend_util_ticks %d\n", t.BackendUtilTicks.Load())
}

// ObserveNode records one macro-node completion duration.
func (t *Telemetry) ObserveNode(d time.Duration) {
	if t == nil {
		return
	}
	t.NodesExecuted.Add(1)
	t.NodeExecNanos.Add(uint64(d.Nanoseconds()))
}

// ObserveBatch records a wavefront batch.
func (t *Telemetry) ObserveBatch() {
	if t == nil {
		return
	}
	t.BatchesExecuted.Add(1)
}

// ObserveSchedulerIdle records idle time in the driver between batches.
func (t *Telemetry) ObserveSchedulerIdle(d time.Duration) {
	if t == nil {
		return
	}
	t.SchedulerIdleNs.Add(uint64(d.Nanoseconds()))
}

// ObserveEdgeWalk records time spent in dependency / edge traversal accounting.
func (t *Telemetry) ObserveEdgeWalk(d time.Duration) {
	if t == nil {
		return
	}
	t.EdgeWalkNanos.Add(uint64(d.Nanoseconds()))
}

// BumpBackendUtil increments an abstract utilization tick (GPU mock or CPU busy).
func (t *Telemetry) BumpBackendUtil(n uint64) {
	if t == nil {
		return
	}
	t.BackendUtilTicks.Add(n)
}
