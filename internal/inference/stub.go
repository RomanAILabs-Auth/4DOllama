package inference

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/ollama"
)

// Stub is the native four_d_engine provider: 4D autoregressive decode (RoPE → attention → projected vocab logits → sample).
type Stub struct{}

func (Stub) Name() string { return "stub" }

const (
	maxCtxRunesAutoreg = 384
	stubSamplerTemp    = float32(0.7)
	stubSamplerTopK    = 40
)

func (Stub) Generate(ctx context.Context, cx Context, model, prompt string) (string, error) {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("model=%s\n", model))
	if cx.ModelPath != "" {
		sb.WriteString(fmt.Sprintf("source=%s\n", cx.ModelPath))
	}
	if cx.ParamCount > 0 {
		sb.WriteString(fmt.Sprintf("gguf_param_count=%d\n", cx.ParamCount))
	}
	if len(cx.LiftedWeights) > 0 {
		sb.WriteString(fmt.Sprintf("lifted_sample_len=%d\n", len(cx.LiftedWeights)))
	}
	if cx.WeightsFrom4DGGUF {
		sb.WriteString("weights_source=.4dgguf\n")
	}
	sb.WriteString(fmt.Sprintf("4d_preflight_gemm_attn_dot_lifted=%.6f\n\n", cx.MatmulScore))
	sb.WriteString(prompt)
	if cx.Eng == nil {
		sb.WriteString("\n\n[4D autoreg unavailable: no engine]\n")
		return sb.String(), nil
	}
	body, nTok, err := autoregressive4D(ctx, cx.Eng, cx.ModelPath, cx.LiftedWeights, prompt)
	if err != nil {
		return "", err
	}
	sb.WriteString("\n\n")
	sb.WriteString(body)
	sb.WriteString(fmt.Sprintf("\n\n(generated_tokens=%d)\n", nTok))
	return sb.String(), nil
}

func runesToFloatsLimited(chain []rune, max int) []float32 {
	if len(chain) > max {
		chain = chain[len(chain)-max:]
	}
	out := make([]float32, 0, len(chain))
	for _, r := range chain {
		out = append(out, float32(r))
	}
	return out
}

func autoregressive4D(ctx context.Context, eng engine.Engine, modelPath string, lifted []float32, prompt string) (string, int, error) {
	if strings.TrimSpace(modelPath) == "" {
		return "", 0, fmt.Errorf("stub autoreg needs a resolved GGUF path on disk")
	}
	tokens, err := LoadGGUFTokenStrings(modelPath)
	if err != nil {
		return "", 0, fmt.Errorf("gguf tokenizer: %w", err)
	}
	vocabSize := len(tokens)
	text := prompt
	if text == "" {
		text = "Hello "
	}
	var dec strings.Builder
	written := 0
	last4 := make([]float32, 4)
	for step := 0; step < AutoregMaxTokens; step++ {
		select {
		case <-ctx.Done():
			return dec.String(), written, ctx.Err()
		default:
		}
		floats := runesToFloatsLimited([]rune(text), maxCtxRunesAutoreg)
		rope, err := eng.Rope4DSequence(floats)
		if err != nil || len(rope) < 4 {
			break
		}
		sl := len(rope) / 4
		attn, aerr := eng.SpacetimeAttention4D(rope, rope, rope, sl)
		copy(last4, rope[len(rope)-4:])
		if aerr == nil && len(attn) >= 4 {
			copy(last4, attn[len(attn)-4:])
		}
		logits, err := eng.ProjectStubLogits(last4, lifted, vocabSize, step, step == 0)
		if err != nil || len(logits) != vocabSize {
			break
		}
		PromptTokenBias(logits, tokens, prompt)
		tid, err := eng.SampleNextTokenFlat(logits, stubSamplerTemp, stubSamplerTopK)
		if err != nil {
			break
		}
		if tid < 0 {
			break
		}
		if tid >= len(tokens) {
			tid = len(tokens) - 1
		}
		piece := tokens[tid]
		if strings.Contains(piece, "</s>") || strings.Contains(piece, "im_end") || strings.TrimSpace(piece) == "<|im_end|>" {
			break
		}
		AppendSPPiece(&dec, piece)
		text = prompt + dec.String()
		written++
	}
	return dec.String(), written, nil
}

// ToResponse wraps text in Ollama JSON shape.
func ToResponse(model, text string) ollama.GenerateResponse {
	return ollama.GenerateResponse{
		Model:     model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Response:  text,
		Done:      true,
	}
}
