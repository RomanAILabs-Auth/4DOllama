package httpserver

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/4dollama/4dollama/internal/ollama"
)

// pullStatusWriter turns line-oriented progress from ollamareg into Ollama-style NDJSON.
type pullStatusWriter struct {
	enc *json.Encoder
	f   http.Flusher
}

func newPullStatusWriter(w http.ResponseWriter) *pullStatusWriter {
	var fl http.Flusher
	if f, ok := w.(http.Flusher); ok {
		fl = f
	}
	return &pullStatusWriter{enc: json.NewEncoder(w), f: fl}
}

func (p *pullStatusWriter) Emit(ev ollama.PullResponse) error {
	if err := p.enc.Encode(ev); err != nil {
		return err
	}
	if p.f != nil {
		p.f.Flush()
	}
	return nil
}

func (p *pullStatusWriter) Write(b []byte) (int, error) {
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		_ = p.enc.Encode(ollama.PullResponse{Status: line})
		if p.f != nil {
			p.f.Flush()
		}
	}
	return len(b), nil
}
