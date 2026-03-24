package httpserver

import (
	"fmt"
	"sync/atomic"
)

// Metrics holds lightweight counters (Prometheus text exposition without extra deps).
type Metrics struct {
	Requests atomic.Uint64
	Errors   atomic.Uint64
}

func (m *Metrics) IncRequest() {
	m.Requests.Add(1)
}

func (m *Metrics) IncError() {
	m.Errors.Add(1)
}

func (m *Metrics) Prometheus() string {
	return fmt.Sprintf(
		"# HELP fourd_http_requests_total Total HTTP requests handled\n# TYPE fourd_http_requests_total counter\nfourd_http_requests_total %d\n# HELP fourd_http_errors_total Total HTTP 4xx/5xx responses\n# TYPE fourd_http_errors_total counter\nfourd_http_errors_total %d\n",
		m.Requests.Load(),
		m.Errors.Load(),
	)
}
