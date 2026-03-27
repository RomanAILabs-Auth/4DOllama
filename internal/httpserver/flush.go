package httpserver

import "net/http"

// flushStreamWriter pushes buffered response bytes to the client immediately.
// Uses http.ResponseController (Go 1.20+) and falls back to http.Flusher for older stacks / wrappers.
func flushStreamWriter(w http.ResponseWriter) {
	if w == nil {
		return
	}
	if err := http.NewResponseController(w).Flush(); err == nil {
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}
