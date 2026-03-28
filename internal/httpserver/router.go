package httpserver

import (
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

const maxRequestBody = 8 << 20 // 8 MiB

// NewRouter wires Ollama/OpenAI-compatible routes.
func NewRouter(h *Handler, log *slog.Logger) http.Handler {
	r := chi.NewRouter()
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if req.ContentLength > maxRequestBody {
				http.Error(w, http.StatusText(http.StatusRequestEntityTooLarge), http.StatusRequestEntityTooLarge)
				return
			}
			req.Body = http.MaxBytesReader(w, req.Body, maxRequestBody)
			next.ServeHTTP(w, req)
		})
	})
	r.Use(middleware.RealIP)
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Cache-Control", "no-store")
			next.ServeHTTP(w, req)
		})
	})
	r.Use(RequestID)
	r.Use(Recover(log))
	r.Use(AccessLog(log))
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			if h.Metrics != nil {
				h.Metrics.IncRequest()
			}
			next.ServeHTTP(w, req)
		})
	})

	r.Get("/healthz", h.Healthz)
	r.Get("/livez", h.Livez)
	r.Get("/metrics", h.MetricsProm)

	r.Get("/api/version", h.Version)
	r.Get("/api/engine", h.Engine)
	r.Get("/api/tags", h.Tags)
	r.Get("/api/ps", h.Ps)
	r.Post("/api/stop", h.Stop)
	r.Post("/api/pull", h.Pull)
	r.Post("/api/generate", h.Generate)
	r.Post("/api/chat", h.Chat)
	r.Post("/api/embeddings", h.Embeddings)

	r.Post("/v1/chat/completions", h.OAIChat)
	r.Post("/v1/completions", h.OAICompletion)

	return r
}
