package httpserver

import (
	"encoding/json"
	"net/http"
	"time"
)

// chunkGenerateResponse splits text into streaming deltas (Ollama-style: client concatenates).
func chunkGenerateResponse(s string, chunkRunes int) []string {
	if chunkRunes < 1 {
		chunkRunes = 1
	}
	// Default for generate (larger chunks); chat passes explicit small chunk for streaming UX.
	if chunkRunes > 1 && chunkRunes < 8 {
		chunkRunes = 24
	}
	r := []rune(s)
	if len(r) == 0 {
		return []string{""}
	}
	var out []string
	for i := 0; i < len(r); i += chunkRunes {
		j := i + chunkRunes
		if j > len(r) {
			j = len(r)
		}
		out = append(out, string(r[i:j]))
	}
	return out
}

func writeOllamaGenerateStream(w http.ResponseWriter, model, createdAt, fullText string, delay time.Duration) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	parts := chunkGenerateResponse(fullText, 28)
	for i, delta := range parts {
		done := i == len(parts)-1
		line, _ := json.Marshal(map[string]any{
			"model":      model,
			"created_at": createdAt,
			"response":   delta,
			"done":       done,
		})
		_, _ = w.Write(append(line, '\n'))
		flushStreamWriter(w)
		if delay > 0 {
			time.Sleep(delay)
		}
	}
}

func writeOllamaChatStream(w http.ResponseWriter, model, createdAt, fullText string, delay time.Duration) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	// One rune (or short grapheme cluster step) per NDJSON line so terminals show smooth streaming.
	parts := chunkGenerateResponse(fullText, 1)
	for i, delta := range parts {
		done := i == len(parts)-1
		line, _ := json.Marshal(map[string]any{
			"model":      model,
			"created_at": createdAt,
			"message":    map[string]string{"role": "assistant", "content": delta},
			"done":       done,
		})
		_, _ = w.Write(append(line, '\n'))
		flushStreamWriter(w)
		if delay > 0 {
			time.Sleep(delay)
		}
	}
}

