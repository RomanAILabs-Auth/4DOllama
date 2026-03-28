package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/models"
)

// cmdShow prints an Ollama-like model summary (path, size, optional GGUF param count).
func cmdShow(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama show <model>[:tag]")
		return 2
	}
	cfg := config.Load()
	reg := models.NewRegistry(cfg.ModelsDir, cfg.OllamaModels, cfg.ShareOllamaBlobs, log)
	name := strings.TrimSpace(args[0])
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	e, ok := reg.Resolve(name)
	if !ok {
		fmt.Fprintf(os.Stderr, "4dollama show: model %q not found\n", name)
		return 1
	}
	format := e.Format
	if format == "" {
		format = "gguf"
	}
	eng := engine.New()
	var pc int64
	if strings.EqualFold(format, "gguf") {
		if n, err := eng.GGUFParamCount(e.Path); err == nil {
			pc = n
		}
	}
	out := map[string]any{
		"name":         name + ":latest",
		"path":         e.Path,
		"size":         e.Size,
		"modified_at":  e.ModifiedAt.Format(time.RFC3339Nano),
		"format":       format,
		"param_count":  pc,
		"engine":       "4dollama-native",
		"lattice_note": "Autoregressive decode couples RoPE + SpacetimeAttention4D → 4D lattice (||QKᵀ||_F proxy) → logit bias",
	}
	if strings.EqualFold(format, "4dai") {
		out["carrier"] = "Cl(4,0) isomorphic 4×4 blocks (romanai.4dai JSON or ROMANAI4+safetensors)"
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
	return 0
}

func cmdRm(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama rm <model>[:tag]")
		return 2
	}
	cfg := config.Load()
	reg := models.NewRegistry(cfg.ModelsDir, cfg.OllamaModels, cfg.ShareOllamaBlobs, log)
	name := strings.TrimSpace(args[0])
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	e, ok := reg.Resolve(name)
	if !ok {
		fmt.Fprintf(os.Stderr, "4dollama rm: model %q not found\n", name)
		return 1
	}
	absModels, _ := filepath.Abs(cfg.ModelsDir)
	absPath, _ := filepath.Abs(e.Path)
	if !strings.HasPrefix(absPath, absModels) {
		fmt.Fprintf(os.Stderr, "4dollama rm: refusing to delete outside FOURD_MODELS (%s); use a copy under your models dir\n", cfg.ModelsDir)
		return 1
	}
	if err := os.Remove(e.Path); err != nil {
		fmt.Fprintf(os.Stderr, "4dollama rm: %v\n", err)
		return 1
	}
	if log != nil {
		log.Info("removed model file", slog.String("path", e.Path))
	}
	fmt.Printf("deleted %s\n", e.Path)
	return 0
}

func cmdCp(args []string, log *slog.Logger) int {
	if len(args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama cp <source> <destination>")
		return 2
	}
	cfg := config.Load()
	reg := models.NewRegistry(cfg.ModelsDir, cfg.OllamaModels, cfg.ShareOllamaBlobs, log)
	srcName := strings.TrimSpace(args[0])
	dstName := strings.TrimSpace(args[1])
	if i := strings.IndexByte(srcName, ':'); i >= 0 {
		srcName = srcName[:i]
	}
	if i := strings.IndexByte(dstName, ':'); i >= 0 {
		dstName = dstName[:i]
	}
	se, ok := reg.Resolve(srcName)
	if !ok {
		fmt.Fprintf(os.Stderr, "4dollama cp: source %q not found\n", srcName)
		return 1
	}
	_ = os.MkdirAll(cfg.ModelsDir, 0o755)
	ext := strings.ToLower(filepath.Ext(se.Path))
	if ext == "" {
		ext = ".gguf"
	}
	dstPath := filepath.Join(cfg.ModelsDir, dstName+ext)
	inF, err := os.Open(se.Path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama cp: %v\n", err)
		return 1
	}
	defer inF.Close()
	outF, err := os.Create(dstPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama cp: %v\n", err)
		return 1
	}
	defer outF.Close()
	if _, err := io.Copy(outF, inF); err != nil {
		fmt.Fprintf(os.Stderr, "4dollama cp: %v\n", err)
		return 1
	}
	if log != nil {
		log.Info("copied model", slog.String("from", se.Path), slog.String("to", dstPath))
	}
	fmt.Printf("copied %s -> %s\n", se.Path, dstPath)
	return 0
}
