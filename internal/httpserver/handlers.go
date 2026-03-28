package httpserver

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/models"
	"github.com/4dollama/4dollama/internal/ollama"
	"github.com/4dollama/4dollama/internal/ollamareg"
	"github.com/4dollama/4dollama/internal/runner"
	"github.com/4dollama/4dollama/internal/version"
)

type Handler struct {
	Run              *runner.Service
	Reg              *models.Registry
	Log              *slog.Logger
	Metrics          *Metrics
	OllamaModels     string        // shared blob store root (~/.ollama/models)
	StreamChunkDelay time.Duration // optional pacing for NDJSON streaming (0 = none)
}

func (h *Handler) Healthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (h *Handler) Livez(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	_, _ = w.Write([]byte("ok\n"))
}

func (h *Handler) MetricsProm(w http.ResponseWriter, _ *http.Request) {
	if h.Metrics == nil {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = io.WriteString(w, h.Metrics.Prometheus())
}

func (h *Handler) Version(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollama.VersionResponse{Version: version.Version})
}

// Engine returns JSON merging the Go server version with the Rust four_d_engine capability manifest.
func (h *Handler) Engine(w http.ResponseWriter, _ *http.Request) {
	var inf engine.Info
	var caps []byte
	var err error
	if h.Run != nil && h.Run.Eng != nil {
		inf = h.Run.Eng.Info()
		caps, err = h.Run.Eng.Capabilities()
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	var capsObj any
	if len(caps) > 0 {
		_ = json.Unmarshal(caps, &capsObj)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"server_version": version.Version,
		"go_backend":     string(inf.Backend),
		"four_d_engine":  capsObj,
	})
}

func (h *Handler) Tags(w http.ResponseWriter, _ *http.Request) {
	list, err := h.Reg.List()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	out := make([]ollama.Model, 0, len(list))
	for _, e := range list {
		format := e.Format
		if format == "" {
			format = "gguf"
		}
		var pc int64
		if strings.EqualFold(format, "gguf") && h.Run != nil && h.Run.Eng != nil {
			pc, _ = h.Run.Eng.GGUFParamCount(e.Path)
		}
		out = append(out, ollama.Model{
			Name:       e.Name + ":latest",
			Model:      e.Name + ":latest",
			ModifiedAt: e.ModifiedAt.Format("2006-01-02T15:04:05.999999999Z07:00"),
			Size:       e.Size,
			Digest:     "sha256:fourd-" + e.Name + "-" + format,
			Details: map[string]any{
				"format":      format,
				"engine":      "4d",
				"param_count": pc,
				"size_bytes":  e.Size,
			},
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollama.TagsResponse{Models: out})
}

func (h *Handler) Ps(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollama.PsResponse{Models: []any{}})
}

// Stop acknowledges a stop request (Ollama-compatible). 4dollama uses stateless generation — no process to kill yet.
func (h *Handler) Stop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	type stopReq struct {
		Name string `json:"name"`
	}
	var req stopReq
	_ = json.NewDecoder(r.Body).Decode(&req)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok":    true,
		"name":  strings.TrimSpace(req.Name),
		"note":  "4dollama: no resident runner; stop is a no-op for API parity",
	})
}

// Pull downloads a model from the Ollama registry into FOURD_MODELS (GGUF layer).
func (h *Handler) Pull(w http.ResponseWriter, r *http.Request) {
	var req ollama.PullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		http.Error(w, "name required", http.StatusBadRequest)
		return
	}
	stream := req.Stream == nil || *req.Stream
	ctx := r.Context()
	dir := h.Reg.Dir()

	if stream {
		w.Header().Set("Content-Type", "application/x-ndjson")
		sw := newPullStatusWriter(w)
		_, err := ollamareg.PullGGUF(ctx, name, dir, h.OllamaModels, sw)
		if err != nil {
			_ = sw.Emit(ollama.PullResponse{Status: "error: " + err.Error()})
			return
		}
		_ = sw.Emit(ollama.PullResponse{Status: "success"})
		return
	}

	_, err := ollamareg.PullGGUF(ctx, name, dir, h.OllamaModels, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollama.PullResponse{Status: "success"})
}

func (h *Handler) Generate(w http.ResponseWriter, r *http.Request) {
	var req ollama.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		http.Error(w, "model required", http.StatusBadRequest)
		return
	}
	stream := req.Stream != nil && *req.Stream
	if stream {
		created := time.Now().UTC().Format(time.RFC3339Nano)
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		if err := h.Run.StreamGenerate(r.Context(), req, func(delta string) error {
			if delta == "" {
				return nil
			}
			line, _ := json.Marshal(map[string]any{
				"model":      req.Model,
				"created_at": created,
				"response":   delta,
				"done":       false,
			})
			_, werr := w.Write(append(line, '\n'))
			flushStreamWriter(w)
			if h.StreamChunkDelay > 0 {
				time.Sleep(h.StreamChunkDelay)
			}
			return werr
		}); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		line, _ := json.Marshal(map[string]any{
			"model":      req.Model,
			"created_at": created,
			"response":   "",
			"done":       true,
		})
		_, _ = w.Write(append(line, '\n'))
		flushStreamWriter(w)
		return
	}
	resp, err := h.Run.Generate(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) Chat(w http.ResponseWriter, r *http.Request) {
	var req ollama.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		http.Error(w, "model required", http.StatusBadRequest)
		return
	}
	stream := req.Stream != nil && *req.Stream
	if stream {
		created := time.Now().UTC().Format(time.RFC3339Nano)
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		if err := h.Run.StreamChat(r.Context(), req, func(delta string) error {
			if delta == "" {
				return nil
			}
			line, _ := json.Marshal(map[string]any{
				"model":      req.Model,
				"created_at": created,
				"message":    map[string]string{"role": "assistant", "content": delta},
				"done":       false,
			})
			_, werr := w.Write(append(line, '\n'))
			flushStreamWriter(w)
			if h.StreamChunkDelay > 0 {
				time.Sleep(h.StreamChunkDelay)
			}
			return werr
		}); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		line, _ := json.Marshal(map[string]any{
			"model":      req.Model,
			"created_at": created,
			"message":    map[string]string{"role": "assistant", "content": ""},
			"done":       true,
		})
		_, _ = w.Write(append(line, '\n'))
		flushStreamWriter(w)
		return
	}
	resp, err := h.Run.Chat(r.Context(), req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) Embeddings(w http.ResponseWriter, r *http.Request) {
	var req ollama.EmbeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	// Deterministic tiny embedding for API parity (not model-quality).
	vec := make([]float32, 8)
	for i, c := range []byte(req.Prompt) {
		vec[i%len(vec)] += float32(c) / 255.0
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollama.EmbeddingsResponse{Embedding: vec})
}

func (h *Handler) OAICompletion(w http.ResponseWriter, r *http.Request) {
	var req ollama.OAICompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	stream := req.Stream != nil && *req.Stream
	if stream {
		streamOn := true
		gr := ollama.GenerateRequest{Model: req.Model, Prompt: req.Prompt, Stream: &streamOn}
		created := time.Now().Unix()
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		if err := h.Run.StreamGenerate(r.Context(), gr, func(delta string) error {
			if delta == "" {
				return nil
			}
			chunk := map[string]any{
				"id":      "cmpl-fourd",
				"object":  "text_completion",
				"created": created,
				"model":   req.Model,
				"choices": []map[string]any{{
					"index": 0,
					"text":  delta,
				}},
			}
			b, _ := json.Marshal(chunk)
			if _, err := w.Write([]byte("data: ")); err != nil {
				return err
			}
			if _, err := w.Write(b); err != nil {
				return err
			}
			if _, err := w.Write([]byte("\n\n")); err != nil {
				return err
			}
			flushStreamWriter(w)
			if h.StreamChunkDelay > 0 {
				time.Sleep(h.StreamChunkDelay)
			}
			return nil
		}); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		last := map[string]any{
			"id": "cmpl-fourd", "object": "text_completion", "created": created, "model": req.Model,
			"choices": []map[string]any{{
				"index": 0, "text": "", "finish_reason": "stop",
			}},
		}
		b, _ := json.Marshal(last)
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\ndata: [DONE]\n\n"))
		flushStreamWriter(w)
		return
	}
	g, err := h.Run.Generate(r.Context(), ollama.GenerateRequest{Model: req.Model, Prompt: req.Prompt, Stream: req.Stream})
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	out := ollama.OAICompletionResponse{
		ID:      "cmpl-fourd",
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ollama.OAICompletionChoice{{
			Text:         g.Response,
			Index:        0,
			FinishReason: "stop",
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (h *Handler) OAIChat(w http.ResponseWriter, r *http.Request) {
	var req ollama.OAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	msgs := make([]ollama.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, ollama.Message{Role: m.Role, Content: m.Content})
	}
	stream := req.Stream != nil && *req.Stream
	if stream {
		streamOn := true
		created := time.Now().Unix()
		id := "chatcmpl-fourd"
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		first := true
		if err := h.Run.StreamChat(r.Context(), ollama.ChatRequest{
			Model: req.Model, Messages: msgs, Stream: &streamOn,
		}, func(delta string) error {
			if delta == "" {
				return nil
			}
			deltaObj := map[string]string{"content": delta}
			if first {
				deltaObj["role"] = "assistant"
				first = false
			}
			chunk := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
				"choices": []map[string]any{{
					"index": 0, "delta": deltaObj, "finish_reason": nil,
				}},
			}
			b, _ := json.Marshal(chunk)
			if _, err := w.Write([]byte("data: ")); err != nil {
				return err
			}
			if _, err := w.Write(b); err != nil {
				return err
			}
			if _, err := w.Write([]byte("\n\n")); err != nil {
				return err
			}
			flushStreamWriter(w)
			if h.StreamChunkDelay > 0 {
				time.Sleep(h.StreamChunkDelay)
			}
			return nil
		}); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		final := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
			"choices": []map[string]any{{
				"index": 0, "delta": map[string]string{}, "finish_reason": "stop",
			}},
		}
		b, _ := json.Marshal(final)
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(b)
		_, _ = w.Write([]byte("\n\ndata: [DONE]\n\n"))
		flushStreamWriter(w)
		return
	}
	cr, err := h.Run.Chat(r.Context(), ollama.ChatRequest{Model: req.Model, Messages: msgs, Stream: req.Stream})
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	out := ollama.OAIChatResponse{
		ID:      "chatcmpl-fourd",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []ollama.OAIChatChoice{{
			Index:        0,
			Message:      ollama.OAIChatMessage{Role: "assistant", Content: cr.Message.Content},
			FinishReason: "stop",
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}
