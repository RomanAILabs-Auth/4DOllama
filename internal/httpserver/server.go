package httpserver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/inference"
	"github.com/4dollama/4dollama/internal/models"
	"github.com/4dollama/4dollama/internal/runner"
	"github.com/4dollama/4dollama/internal/version"
)

// Run boots the HTTP server until SIGINT/SIGTERM.
func Run(ctx context.Context, cfg config.Config, log *slog.Logger, fourDMode bool) error {
	inf, err := inference.NewFromConfig(cfg)
	if err != nil {
		return err
	}
	eng := engine.New()
	reg := models.NewRegistry(cfg.ModelsDir, cfg.OllamaModels, cfg.ShareOllamaBlobs, log)
	svc := runner.NewService(eng, reg, log, inf)
	met := &Metrics{}
	chunkDelay := time.Duration(cfg.StreamChunkMs) * time.Millisecond
	h := &Handler{
		Run: svc, Reg: reg, Log: log, FourD: fourDMode, Metrics: met,
		OllamaModels: cfg.OllamaModels, StreamChunkDelay: chunkDelay,
	}

	srv := &http.Server{
		Addr:              cfg.Addr(),
		Handler:           NewRouter(h, log),
		ReadHeaderTimeout: cfg.HTTPRead,
		ReadTimeout:       cfg.HTTPRead,
		WriteTimeout:      cfg.HTTPWrite,
		IdleTimeout:       cfg.HTTPIdle,
		BaseContext:       func(_ net.Listener) context.Context { return ctx },
	}

	errCh := make(chan error, 1)
	go func() {
		// One line on stderr, Ollama-style; details are debug-only (serve -verbose / FOURD_LOG_LEVEL=debug).
		fmt.Fprintf(os.Stderr, "Listening on http://%s (version %s)\n", cfg.Addr(), version.Version)
		log.Debug("serve config",
			slog.String("addr", cfg.Addr()),
			slog.String("models", cfg.ModelsDir),
			slog.String("ollama_models", cfg.OllamaModels),
			slog.Bool("share_ollama_blobs", cfg.ShareOllamaBlobs),
			slog.String("inference", inf.Name()),
		)
		err := srv.ListenAndServe()
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-sig:
		log.Debug("shutdown signal")
	case err := <-errCh:
		return err
	case <-ctx.Done():
		log.Debug("context cancelled")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	return <-errCh
}
