package main

import (
	"flag"
	"log/slog"
	"os"

	"github.com/4dollama/4dollama/internal/cli"
	"github.com/4dollama/4dollama/internal/config"
)

func main() {
	help := flag.Bool("help", false, "show help")
	mode4d := flag.Bool("fourd-mode", false, "enable true 4D native inference path (higher memory)")
	flag.Parse()

	log := newLogger(config.Load())

	if *help {
		cli.Usage()
		return
	}

	args := flag.Args()
	if len(args) == 0 {
		cli.Usage()
		os.Exit(2)
	}

	os.Exit(cli.Run(args, log, *mode4d))
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
