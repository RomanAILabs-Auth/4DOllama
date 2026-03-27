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
	err := (Stub{}).GenerateStream(ctx, cx, model, prompt, func(s string) error {
		sb.WriteString(s)
		return nil
	})
	return sb.String(), err
}

// GenerateStream streams native 4D autoregressive tokens as they are sampled (plus the same preamble/footer as Generate).
func (Stub) GenerateStream(ctx context.Context, cx Context, model, prompt string, emit func(string) error) error {
	if emit == nil {
		emit = func(string) error { return nil }
	}
	e := func(s string) error {
		if s == "" {
			return nil
		}
		return emit(s)
	}
	if err := e(fmt.Sprintf("model=%s\n", model)); err != nil {
		return err
	}
	if cx.ModelPath != "" {
		if err := e(fmt.Sprintf("source=%s\n", cx.ModelPath)); err != nil {
			return err
		}
	}
	if cx.ParamCount > 0 {
		if err := e(fmt.Sprintf("gguf_param_count=%d\n", cx.ParamCount)); err != nil {
			return err
		}
	}
	if len(cx.LiftedWeights) > 0 {
		if err := e(fmt.Sprintf("lifted_sample_len=%d\n", len(cx.LiftedWeights))); err != nil {
			return err
		}
	}
	if cx.WeightsFrom4DGGUF {
		if err := e("weights_source=.4dgguf\n"); err != nil {
			return err
		}
	}
	if err := e(fmt.Sprintf("4d_preflight_gemm_attn_dot_lifted=%.6f\n\n", cx.MatmulScore)); err != nil {
		return err
	}
	if err := e(prompt); err != nil {
		return err
	}
	if cx.Eng == nil {
		return e("\n\n[4D autoreg unavailable: no engine]\n")
	}
	if err := e("\n\n"); err != nil {
		return err
	}
	nTok, err := autoregressive4DStream(ctx, cx.Eng, cx.ModelPath, cx.LiftedWeights, prompt, emit)
	if err != nil {
		return err
	}
	return e(fmt.Sprintf("\n\n(generated_tokens=%d)\n", nTok))
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

// autoregressive4DStream samples tokens and calls emit with each decoded delta (SentencePiece-aware).
func autoregressive4DStream(ctx context.Context, eng engine.Engine, modelPath string, lifted []float32, prompt string, emit func(string) error) (int, error) {
	if strings.TrimSpace(modelPath) == "" {
		return 0, fmt.Errorf("stub autoreg needs a resolved GGUF path on disk")
	}
	tokens, err := LoadGGUFTokenStrings(modelPath)
	if err != nil {
		return 0, fmt.Errorf("gguf tokenizer: %w", err)
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
			return written, ctx.Err()
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
		// Native 4D lattice ↔ QK proxy: RoPE + spacetime attention drive cognitive gravity; lattice biases logits.
		latBias := GlobalLattice().OnTokenStep(rope, attn, lifted, step)
		ApplyLogitBias(logits, latBias)
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
		prevLen := dec.Len()
		AppendSPPiece(&dec, piece)
		if emit != nil && dec.Len() > prevLen {
			if err := emit(dec.String()[prevLen:]); err != nil {
				return written, err
			}
		}
		text = prompt + dec.String()
		written++
	}
	return written, nil
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
