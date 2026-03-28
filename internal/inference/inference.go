// Package inference selects how /api/generate is fulfilled: native four_d_engine 4D decode (stub)
// vs optional upstream Ollama (hybrid) when FOURD_INFERENCE=ollama.
package inference

import (
	"context"
	"fmt"
	"strings"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/engine"
)

// AutoregMaxTokens is the stub 4D autoregressive decode horizon (RoPE → attention → FFN → sample per step).
const AutoregMaxTokens = 128

// Context is metadata gathered by the runner before invoking a provider.
type Context struct {
	ModelResolved bool
	ModelPath     string
	InspectJSON   string
	Eng           engine.Engine
	FourDDemo     []float32 // quaternion demo output from engine.Compute4DDemo
	LiftedWeights []float32 // GGUF sample after 4D lift (native)
	ParamCount    int64     // GGUF parameter element count
	RoPEEmbedding []float32 // prompt embeddings after quaternion RoPE (per-token quads)
	// SpacetimeAttention is causal 4D quaternion attention over RoPEEmbedding (same seq).
	SpacetimeAttention []float32
	MatmulScore        float32 // attention_out · lifted weights (when attention ok), else RoPE · lifted
	WeightsFrom4DGGUF  bool    // lifted tensor cache loaded from ~/.ollama/models/blobs/<model>.4dgguf
	// ShardPaths lists native .4dai blob paths when the model is registered via .multi4dai (optional).
	ShardPaths []string
	// TokenizerGGUF is optional: read tokenizer.ggml.tokens from this GGUF for .4dai stub decode.
	TokenizerGGUF string
}

// Provider implements completion for a single backend strategy.
type Provider interface {
	Name() string
	Generate(ctx context.Context, reqCtx Context, model, prompt string) (text string, err error)
}

// NewFromConfig returns the configured provider or an error if settings are inconsistent.
func NewFromConfig(cfg config.Config) (Provider, error) {
	mode := strings.ToLower(strings.TrimSpace(cfg.InferenceMode))
	switch mode {
	case "", "stub", "local", "demo", "fourd", "4d", "native", "engine":
		return Stub{}, nil
	case "ollama", "forward", "upstream":
		if cfg.OllamaHost == "" {
			return nil, fmt.Errorf("FOURD_INFERENCE=%q requires OLLAMA_HOST (e.g. http://127.0.0.1:11434)", cfg.InferenceMode)
		}
		return OllamaForward{BaseURL: strings.TrimSuffix(cfg.OllamaHost, "/")}, nil
	default:
		return nil, fmt.Errorf("unknown FOURD_INFERENCE %q (use stub|fourd for native 4D engine, or ollama for hybrid)", cfg.InferenceMode)
	}
}
