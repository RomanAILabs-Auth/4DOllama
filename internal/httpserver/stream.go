package httpserver

import (
	"encoding/json"
	"net/http"
	"time"
)

// chunkGenerateResponse splits text into streaming deltas (Ollama-style: client concatenates).
func chunkGenerateResponse(s string, chunkRunes int) []string {
	if chunkRunes < 8 {
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
	fl, _ := w.(http.Flusher)
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
		if fl != nil {
			fl.Flush()
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	}
}

func writeOllamaChatStream(w http.ResponseWriter, model, createdAt, fullText string, delay time.Duration) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	fl, _ := w.(http.Flusher)
	parts := chunkGenerateResponse(fullText, 28)
	for i, delta := range parts {
		done := i == len(parts)-1
		line, _ := json.Marshal(map[string]any{
			"model":      model,
			"created_at": createdAt,
			"message":    map[string]string{"role": "assistant", "content": delta},
			"done":       done,
		})
		_, _ = w.Write(append(line, '\n'))
		if fl != nil {
			fl.Flush()
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	}
}

func writeOAIChatSSE(w http.ResponseWriter, model, id string, created int64, fullText string, delay time.Duration) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	parts := chunkGenerateResponse(fullText, 24)
	for i, delta := range parts {
		deltaObj := map[string]string{"content": delta}
		if i == 0 {
			deltaObj["role"] = "assistant"
		}
		finish := any(nil)
		if i == len(parts)-1 {
			finish = "stop"
		}
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         deltaObj,
				"finish_reason": finish,
			}},
		}
		b, _ := json.Marshal(chunk)
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\n"))
		if fl != nil {
			fl.Flush()
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if fl != nil {
		fl.Flush()
	}
}

func writeOAICompletionSSE(w http.ResponseWriter, model, id string, created int64, fullText string, delay time.Duration) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	parts := chunkGenerateResponse(fullText, 24)
	for i, delta := range parts {
		finish := any(nil)
		if i == len(parts)-1 {
			finish = "stop"
		}
		chunk := map[string]any{
			"id":      id,
			"object":  "text_completion",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index": 0,
				"text":  delta,
			}},
		}
		if i == len(parts)-1 {
			chunk["choices"].([]map[string]any)[0]["finish_reason"] = finish
		}
		b, _ := json.Marshal(chunk)
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\n"))
		if fl != nil {
			fl.Flush()
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if fl != nil {
		fl.Flush()
	}
}
