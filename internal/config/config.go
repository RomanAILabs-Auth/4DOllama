package config

import (
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// DefaultListenPort is the built-in HTTP port (product default; Ollama uses 11434 — set FOURD_PORT=11434 to share that socket).
const DefaultListenPort = "13377"

// Config holds process-wide settings (env-driven for 12-factor deployments).
type Config struct {
	Host       string
	Port       string
	ModelsDir  string
	LogLevel   slog.Level
	LogJSON    bool
	HTTPRead   time.Duration
	HTTPWrite  time.Duration
	HTTPIdle   time.Duration
	OllamaHost        string // for benchmark-4d baseline
	OllamaModels      string // ~/.ollama/models — manifests + blobs
	ShareOllamaBlobs  bool   // list/resolve/pull reuse Ollama's blob store
	DefaultTestModel  string // hint only (e.g. qwen2.5) — docs / install messaging
	InferenceMode     string // stub (native four_d_engine 4D decode) | ollama (optional upstream hybrid)
	StreamChunkMs     int    // artificial delay between NDJSON chunks (0 = none)
}

func getenv(key, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}

func parseBool(s string, def bool) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}

func getenvInt(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Load reads configuration from the environment with sensible defaults.
func Load() Config {
	home, _ := os.UserHomeDir()
	defModels := filepath.Join(home, ".4dollama", "models")
	defOllama := filepath.Join(home, ".ollama", "models")
	ollamaHost := strings.TrimSuffix(getenv("OLLAMA_HOST", ""), "/")
	inf := strings.TrimSpace(os.Getenv("FOURD_INFERENCE"))
	if inf == "" {
		// Default: native four_d_engine 4D autoregressive decode on pulled GGUF (no hybrid).
		// Opt-in to upstream llama.cpp via Ollama: FOURD_INFERENCE=ollama and OLLAMA_HOST=...
		inf = "stub"
	}
	logLvl := strings.TrimSpace(os.Getenv("FOURD_LOG_LEVEL"))
	if logLvl == "" {
		logLvl = strings.TrimSpace(os.Getenv("LOG_LEVEL"))
	}
	if logLvl == "" {
		// Quiet default: matches Ollama-style CLI (no per-request or engine spam on stderr).
		logLvl = "warn"
	}
	host := getenv("FOURD_HOST", "0.0.0.0")
	port := getenv("FOURD_PORT", DefaultListenPort)
	// FOURD_DROPIN=1: prefer loopback when FOURD_HOST is unset (local “drop-in” feel). Port stays DefaultListenPort unless FOURD_PORT is set (use 11434 to replace Ollama on its socket).
	if parseBool(getenv("FOURD_DROPIN", "false"), false) {
		if _, set := os.LookupEnv("FOURD_HOST"); !set {
			host = "127.0.0.1"
		}
	}
	return Config{
		Host:             host,
		Port:             port,
		ModelsDir:        getenv("FOURD_MODELS", defModels),
		LogLevel:         parseLevel(logLvl),
		LogJSON:          strings.EqualFold(getenv("FOURD_LOG_JSON", "false"), "true"),
		HTTPRead:         30 * time.Second,
		HTTPWrite:        0, // no write timeout for streaming endpoints
		HTTPIdle:         120 * time.Second,
		OllamaHost:       ollamaHost,
		OllamaModels:     getenv("OLLAMA_MODELS", defOllama),
		ShareOllamaBlobs: parseBool(getenv("FOURD_SHARE_OLLAMA", "true"), true),
		DefaultTestModel: getenv("FOURD_DEFAULT_MODEL", "qwen2.5"),
		InferenceMode:    inf,
		StreamChunkMs:    getenvInt("FOURD_STREAM_CHUNK_MS", 0),
	}
}

// Addr returns host:port for net.Listen.
func (c Config) Addr() string {
	return net.JoinHostPort(c.Host, c.Port)
}

// MustAtoi is used for CLI flags overriding port.
func MustAtoi(s string, def int) int {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return def
	}
	return n
}
