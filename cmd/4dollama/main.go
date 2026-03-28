package main

import (
	"log/slog"
	"os"

	"github.com/4dollama/4dollama/internal/cli"
	"github.com/4dollama/4dollama/internal/config"
)

func main() {
	log := newLogger(config.Load())
	os.Exit(cli.Execute(log))
}

func newLogger(cfg config.Config) *slog.Logger {
	opts := &slog.HandlerOptions{Level: cfg.LogLevel}
	var h slog.Handler
	if cfg.LogJSON {
		h = slog.NewJSONHandler(os.Stderr, opts)
	} else {
		h = slog.NewTextHandler(os.Stderr, opts)
	}
	return slog.New(h)
}
