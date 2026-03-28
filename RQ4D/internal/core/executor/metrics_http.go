package executor

import "net/http"

// MetricsHandler returns an http.Handler that writes Prometheus text from tel (bind on 127.0.0.1 only at call site).
func MetricsHandler(tel *Telemetry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		tel.WritePrometheusText(w)
	})
}
