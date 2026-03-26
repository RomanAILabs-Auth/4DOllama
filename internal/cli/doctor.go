package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/version"
)

// detectDoctorGPU mirrors engine GPU heuristics for the CLI (no CGO required).
func detectDoctorGPU() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FOURD_GPU"))) {
	case "0", "off", "cpu", "none":
		return "cpu"
	}
	if runtime.GOOS == "darwin" {
		if st, err := os.Stat("/System/Library/Frameworks/Metal.framework"); err == nil && st.IsDir() {
			return "metal"
		}
	}
	for _, p := range []string{
		"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
		"/usr/lib/wsl/lib/libcuda.so",
	} {
		if _, err := os.Stat(p); err == nil {
			return "cuda"
		}
	}
	if runtime.GOOS == "windows" && strings.TrimSpace(os.Getenv("CUDA_PATH")) != "" {
		return "cuda"
	}
	return "cpu"
}

func cmdDoctor(log *slog.Logger) int {
	_ = log
	cfg := config.Load()
	gpu := detectDoctorGPU()

	fmt.Println("4dollama doctor")
	fmt.Println(strings.Repeat("-", 48))
	fmt.Printf("version:     %s\n", version.Version)
	fmt.Printf("go_runtime:  %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	fmt.Printf("listen:      %s:%s\n", cfg.Host, cfg.Port)
	fmt.Printf("models_dir:  %s\n", cfg.ModelsDir)
	fmt.Printf("inference:   %s\n", cfg.InferenceMode)
	if cfg.OllamaHost != "" {
		fmt.Printf("OLLAMA_HOST: %s\n", cfg.OllamaHost)
	}
	fmt.Printf("FOURD_GPU:   %q\n", strings.TrimSpace(os.Getenv("FOURD_GPU")))

	if gpu == "cpu" {
		fmt.Println()
		fmt.Println("✓ CPU mode active — no CUDA/Metal acceleration path detected (expected on CPU-only machines).")
		fmt.Println("  Full 4D pipeline (RoPE, attention, GEMM, autoreg) runs on CPU with all engine logs.")
	} else {
		fmt.Println()
		fmt.Printf("✓ GPU-style parallel path: %s (FOURD_GPU=cpu forces CPU-only.)\n", gpu)
	}

	base := baseURL()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(base + "/healthz")
	if err != nil {
		fmt.Println()
		fmt.Printf("✗ API not reachable at %s — run: 4dollama serve\n", base)
		return 1
	}
	resp.Body.Close()
	fmt.Printf("\n✓ API healthy: %s\n", base)

	if !strings.EqualFold(cfg.InferenceMode, "ollama") {
		fmt.Println()
		fmt.Println("✓ Inference: native four_d_engine (FOURD_INFERENCE=stub default) — pulled GGUF is decoded on the 4D path.")
	}
	if strings.EqualFold(cfg.InferenceMode, "ollama") && cfg.OllamaHost != "" {
		oh := strings.TrimSuffix(cfg.OllamaHost, "/")
		if r, err := client.Get(oh + "/api/tags"); err == nil {
			r.Body.Close()
			fmt.Printf("✓ Hybrid: Ollama reachable at OLLAMA_HOST (FOURD_INFERENCE=ollama)\n")
		} else {
			fmt.Printf("✗ OLLAMA_HOST not reachable at %s — start `ollama serve` or unset hybrid (default native 4D)\n", oh)
		}
	}

	if r, err := client.Get(base + "/api/tags"); err == nil {
		b, _ := io.ReadAll(r.Body)
		r.Body.Close()
		var tags struct {
			Models []struct {
				Name string `json:"name"`
			} `json:"models"`
		}
		_ = json.Unmarshal(b, &tags)
		fmt.Printf("✓ models listed: %d (try: 4dollama run %s)\n", len(tags.Models), cfg.DefaultTestModel)
	}

	if r, err := client.Get(base + "/api/engine"); err == nil {
		b, _ := io.ReadAll(r.Body)
		r.Body.Close()
		var root map[string]any
		if json.Unmarshal(b, &root) == nil {
			if fd, ok := root["four_d_engine"].(map[string]any); ok {
				if ffi, ok := fd["ffi"].(map[string]any); ok {
					if gb, ok := ffi["gpu_backend"].(string); ok {
						fmt.Printf("✓ engine reports gpu_backend=%q\n", gb)
					}
				}
			}
		}
	}

	return 0
}
