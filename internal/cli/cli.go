package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/httpserver"
	"github.com/4dollama/4dollama/internal/ollamareg"
	"github.com/4dollama/4dollama/internal/tui"
	"github.com/4dollama/4dollama/internal/version"
	"golang.org/x/term"
)

// Usage prints CLI help to stdout.
func Usage() {
	fmt.Print(`4dollama — Ollama-compatible surface with the 4D engine

Usage:
  4dollama [global flags] <command> [args]

Commands:
  serve              Start the HTTP API (default :13373, avoids Ollama on :11434)
                       Use serve -verbose for debug-level engine logs (same as FOURD_LOG_LEVEL=debug)
  pull <model>       Download GGUF from Ollama registry (registry.ollama.ai)
  import-ollama <m>  Copy GGUF from local Ollama after "ollama pull" (tensor-safe path)
  run <model>        Ollama-style >>> REPL (line mode); optional FOURD_TUI=1 for full-screen TUI
  list               List models (GET /api/tags)
  ps                 Running models (stub)
  version            Print version
  benchmark-4d       Compare latency vs OLLAMA_HOST (if set)
  benchmark          Table: tokens/sec + coherence vs Ollama (same model)
  doctor             Health, GPU (CPU vs CUDA/Metal), models — confirms CPU mode when no GPU

Global flags:
  -help
  -fourd-mode       Enable 4D coherence features in responses

Environment:
  FOURD_HOST, FOURD_PORT (default 13373), FOURD_MODELS, FOURD_LOG_LEVEL (or LOG_LEVEL), FOURD_LOG_JSON
  FOURD_TUI=1         Use bubble-tea full-screen chat instead of Ollama-style >>> line REPL
  FOURD_LINE_CHAT=1    Force line REPL (default is already line mode)
  FOURD_INFERENCE — ollama when OLLAMA_HOST is set, else stub; set stub explicitly for demo tokens only
  FOURD_STREAM_CHUNK_MS — optional delay between streamed response chunks (default 0)
  OLLAMA_HOST, OLLAMA_REGISTRY, OLLAMA_MODELS
  FOURD_SHARE_OLLAMA (default true) — list/resolve/pull reuse ~/.ollama/models blobs
  FOURD_DEFAULT_MODEL — optional hint (default qwen2.5)

`)
}

// LoggerFromConfig builds stderr slog (same shape as main); used by serve -verbose.
func LoggerFromConfig(cfg config.Config) *slog.Logger {
	opts := &slog.HandlerOptions{Level: cfg.LogLevel}
	var h slog.Handler
	if cfg.LogJSON {
		h = slog.NewJSONHandler(os.Stderr, opts)
	} else {
		h = slog.NewTextHandler(os.Stderr, opts)
	}
	return slog.New(h)
}

// Run parses argv (after global flags) and dispatches subcommands.
func Run(args []string, log *slog.Logger, fourDMode bool) int {
	if len(args) == 0 {
		Usage()
		return 2
	}
	cmd := args[0]
	rest := args[1:]
	switch cmd {
	case "serve":
		return cmdServe(rest, log, fourDMode)
	case "pull":
		return cmdPull(rest, log)
	case "import-ollama":
		return cmdImportOllama(rest, log)
	case "run":
		return cmdRun(rest, log, fourDMode)
	case "list":
		return cmdList(rest, log)
	case "ps":
		return cmdPs(rest, log)
	case "version", "--version", "-v":
		fmt.Println(version.Version)
		return 0
	case "create", "show", "rm", "cp", "export", "import", "push", "stop":
		fmt.Fprintf(os.Stderr, "4dollama: %q is reserved for Ollama parity — not implemented in this release.\n", cmd)
		return 2
	case "benchmark-4d":
		return cmdBenchmark(rest, log)
	case "benchmark":
		return cmdBenchmarkTable(rest, log)
	case "doctor":
		return cmdDoctor(log)
	default:
		fmt.Fprintf(os.Stderr, "4dollama: unknown command %q\n", cmd)
		Usage()
		return 2
	}
}

func cmdPull(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama pull <model>[:tag]")
		return 2
	}
	cfg := config.Load()
	ctx := context.Background()
	name := strings.TrimSpace(args[0])
	path, err := ollamareg.PullGGUF(ctx, name, cfg.ModelsDir, cfg.OllamaModels, os.Stderr)
	if err != nil {
		log.Error("pull failed", slog.String("model", name), slog.Any("err", err))
		fmt.Fprintf(os.Stderr, "pull failed: %v\n", err)
		return 1
	}
	fmt.Println(path)
	log.Info("pull complete", slog.String("path", path))
	return 0
}

func cmdImportOllama(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama import-ollama <model>[:tag]")
		return 2
	}
	cfg := config.Load()
	name := strings.TrimSpace(args[0])
	path, err := ollamareg.ImportFromOllamaHome(name, cfg.ModelsDir, cfg.OllamaModels, os.Stderr)
	if err != nil {
		log.Error("import failed", slog.String("model", name), slog.Any("err", err))
		fmt.Fprintf(os.Stderr, "import failed: %v\n", err)
		return 1
	}
	fmt.Println(path)
	log.Info("import complete", slog.String("path", path))
	return 0
}

func cmdServe(args []string, _ *slog.Logger, four bool) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	verbose := fs.Bool("verbose", false, "debug-level logs (4D engine diagnostics on stderr)")
	fs.SetOutput(io.Discard)
	_ = fs.Parse(args)
	cfg := config.Load()
	if *verbose {
		cfg.LogLevel = slog.LevelDebug
	}
	log := LoggerFromConfig(cfg)
	ctx := context.Background()
	if err := httpserver.Run(ctx, cfg, log, four); err != nil {
		log.Error("server stopped", slog.Any("err", err))
		return 1
	}
	return 0
}

func baseURL() string {
	cfg := config.Load()
	host := cfg.Host
	if host == "0.0.0.0" {
		host = "127.0.0.1"
	}
	return "http://" + host + ":" + cfg.Port
}

func cmdRun(args []string, log *slog.Logger, fourD bool) int {
	_ = fourD
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama run <model> [prompt...]")
		return 2
	}
	model := args[0]
	stdoutTTY := term.IsTerminal(int(os.Stdout.Fd()))
	interactiveOut := len(args) == 1 && stdoutTTY
	if err := ensureServerRunning(log, interactiveOut); err != nil {
		fmt.Fprintf(os.Stderr, "4dollama run: %v\n", err)
		return 1
	}
	if len(args) == 1 {
		stdinTTY := term.IsTerminal(int(os.Stdin.Fd()))
		forceLine := strings.TrimSpace(os.Getenv("FOURD_LINE_CHAT")) == "1" ||
			strings.EqualFold(strings.TrimSpace(os.Getenv("FOURD_NO_TUI")), "1")
		useBubble := (strings.TrimSpace(os.Getenv("FOURD_TUI")) == "1" ||
			strings.EqualFold(strings.TrimSpace(os.Getenv("FOURD_TUI")), "true")) && stdinTTY

		// Non-interactive: no TTY on stdout (pipe/redirect) — one-shot from stdin or fail.
		if !stdoutTTY {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				log.Error("read stdin", slog.Any("err", err))
				return 1
			}
			prompt := strings.TrimSpace(string(b))
			return cmdRunGenerate(model, prompt)
		}

		// Interactive: default Ollama-style linear REPL (>>> + line in/out). Bubble full-screen UI
		// only when FOURD_TUI=1 and stdin is a real TTY.
		base := baseURL()
		if forceLine || !useBubble {
			if err := tui.RunLineChat(model, base); err != nil {
				fmt.Fprintf(os.Stderr, "4dollama run: %v\n", err)
				return 1
			}
			return 0
		}
		if err := tui.RunInteractive(model, base); err != nil {
			fmt.Fprintf(os.Stderr, "4dollama run: TUI exited (%v); falling back to line mode\n", err)
			if err2 := tui.RunLineChat(model, base); err2 != nil {
				fmt.Fprintf(os.Stderr, "4dollama run: %v\n", err2)
				return 1
			}
		}
		return 0
	}
	prompt := strings.TrimSpace(strings.Join(args[1:], " "))
	return cmdRunGenerate(model, prompt)
}

func cmdRunGenerate(model, prompt string) int {
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "4dollama run: empty prompt (pipe text on stdin or pass prompt args)")
		return 2
	}
	body := map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	}
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest(http.MethodPost, baseURL()+"/api/generate", bytes.NewReader(buf))
	if err != nil {
		return 1
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama run: %v (is the server running? try `4dollama serve`)\n", err)
		return 1
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		fmt.Fprintf(os.Stderr, "%s\n", strings.TrimSpace(string(b)))
		return 1
	}
	var out struct {
		Response string `json:"response"`
	}
	_ = json.Unmarshal(b, &out)
	fmt.Print(out.Response)
	if !strings.HasSuffix(out.Response, "\n") {
		fmt.Println()
	}
	return 0
}

func cmdList(_ []string, _ *slog.Logger) int {
	req, err := http.NewRequest(http.MethodGet, baseURL()+"/api/tags", nil)
	if err != nil {
		return 1
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama list: %v\n", err)
		return 1
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		fmt.Fprintf(os.Stderr, "%s\n", string(b))
		return 1
	}
	var tags struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	_ = json.Unmarshal(b, &tags)
	for _, m := range tags.Models {
		fmt.Printf("%s\t%d\n", m.Name, m.Size)
	}
	return 0
}

func cmdPs(_ []string, _ *slog.Logger) int {
	req, _ := http.NewRequest(http.MethodGet, baseURL()+"/api/ps", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama ps: %v\n", err)
		return 1
	}
	defer resp.Body.Close()
	_, _ = io.Copy(os.Stdout, resp.Body)
	fmt.Println()
	return 0
}

func pickBenchModel() string {
	req, _ := http.NewRequest(http.MethodGet, baseURL()+"/api/tags", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	_ = json.Unmarshal(b, &tags)
	if len(tags.Models) == 0 {
		return ""
	}
	name := tags.Models[0].Name
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	return name
}

func cmdBenchmark(_ []string, log *slog.Logger) int {
	cfg := config.Load()
	model := os.Getenv("FOURD_BENCH_MODEL")
	if model == "" {
		model = pickBenchModel()
	}
	if model == "" {
		model = "_none_"
	}
	payload, _ := json.Marshal(map[string]any{
		"model":  model,
		"prompt": "benchmark ping",
		"stream": false,
	})
	do := func(base string) (time.Duration, int, error) {
		start := time.Now()
		req, err := http.NewRequest(http.MethodPost, strings.TrimSuffix(base, "/")+"/api/generate", bytes.NewReader(payload))
		if err != nil {
			return 0, 0, err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, 0, err
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		return time.Since(start), resp.StatusCode, nil
	}
	u4 := baseURL()
	d1, c1, err := do(u4)
	if err != nil {
		log.Error("benchmark fourd failed", slog.Any("err", err))
		return 1
	}
	fmt.Printf("4dollama %s: %v status=%d model=%q\n", u4, d1, c1, model)
	if cfg.OllamaHost == "" {
		fmt.Println("OLLAMA_HOST not set — skipping baseline Ollama comparison.")
		return 0
	}
	d2, c2, err := do(cfg.OllamaHost)
	if err != nil {
		log.Error("benchmark ollama failed", slog.Any("err", err))
		return 1
	}
	fmt.Printf("ollama %s: %v status=%d\n", cfg.OllamaHost, d2, c2)
	if d2 > 0 {
		fmt.Printf("ratio (fourd/ollama): %.3fx\n", float64(d1)/float64(d2))
	}
	return 0
}

func benchGenerateAPI(baseURL, model, prompt string) (elapsed time.Duration, responseText string, statusCode int, err error) {
	payload, _ := json.Marshal(map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	})
	start := time.Now()
	req, err := http.NewRequest(http.MethodPost, strings.TrimSuffix(baseURL, "/")+"/api/generate", bytes.NewReader(payload))
	if err != nil {
		return 0, "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, "", 0, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	elapsed = time.Since(start)
	var out struct {
		Response string `json:"response"`
	}
	_ = json.Unmarshal(body, &out)
	return elapsed, out.Response, resp.StatusCode, nil
}

func coherenceScore(s string) float64 {
	if s == "" {
		return 0
	}
	n := utf8.RuneCountInString(s)
	if n == 0 {
		return 0
	}
	seen := make(map[rune]struct{})
	for _, r := range s {
		seen[r] = struct{}{}
	}
	ratio := float64(len(seen)) / float64(n)
	return ratio * math.Min(1.0, float64(n)/120.0)
}

func cmdBenchmarkTable(_ []string, log *slog.Logger) int {
	cfg := config.Load()
	model := os.Getenv("FOURD_BENCH_MODEL")
	if model == "" {
		model = pickBenchModel()
	}
	if model == "" {
		model = cfg.DefaultTestModel
	}
	prompt := os.Getenv("FOURD_BENCH_PROMPT")
	if prompt == "" {
		prompt = "Hello 4D world"
	}
	u4 := baseURL()
	d4, text4, st4, err := benchGenerateAPI(u4, model, prompt)
	if err != nil {
		log.Error("benchmark fourd failed", slog.Any("err", err))
		return 1
	}
	tok4 := utf8.RuneCountInString(text4)
	sec4 := d4.Seconds()
	tps4 := 0.0
	if sec4 > 0 {
		tps4 = float64(tok4) / sec4
	}
	co4 := coherenceScore(text4)
	fmt.Printf("%-14s %10s %12s %14s %12s\n", "backend", "latency", "tokens", "tokens/sec", "coherence")
	fmt.Printf("%-14s %10v %12d %14.2f %12.4f\n", "4dollama", d4.Truncate(time.Millisecond), tok4, tps4, co4)
	fmt.Printf("%-14s status=%d model=%q prompt=%q\n", "", st4, model, prompt)
	if cfg.OllamaHost == "" {
		fmt.Println("OLLAMA_HOST not set — Ollama row skipped.")
		return 0
	}
	do, to, sto, err := benchGenerateAPI(cfg.OllamaHost, model, prompt)
	if err != nil {
		log.Error("benchmark ollama failed", slog.Any("err", err))
		return 1
	}
	toko := utf8.RuneCountInString(to)
	seco := do.Seconds()
	tpso := 0.0
	if seco > 0 {
		tpso = float64(toko) / seco
	}
	coo := coherenceScore(to)
	fmt.Printf("%-14s %10v %12d %14.2f %12.4f\n", "ollama", do.Truncate(time.Millisecond), toko, tpso, coo)
	fmt.Printf("%-14s status=%d\n", "", sto)
	return 0
}
