package runner

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/inference"
	"github.com/4dollama/4dollama/internal/models"
	"github.com/4dollama/4dollama/internal/ollama"
)

const demoPromptMaxRunes = 256

func promptRunesToFloats(prompt string) []float32 {
	if prompt == "" {
		return nil
	}
	var out []float32
	for _, r := range prompt {
		if len(out) >= demoPromptMaxRunes {
			break
		}
		out = append(out, float32(r))
	}
	return out
}

// Service orchestrates model resolution, GGUF inspect, and the inference provider.
type Service struct {
	Eng      engine.Engine
	Registry *models.Registry
	Log      *slog.Logger
	Infer    inference.Provider
}

// NewService wires the runner. If inf is nil, Stub is used.
func NewService(eng engine.Engine, reg *models.Registry, log *slog.Logger, inf inference.Provider) *Service {
	if inf == nil {
		inf = inference.Stub{}
	}
	return &Service{Eng: eng, Registry: reg, Log: log, Infer: inf}
}

// buildInferContext resolves the model and gathers tensors / engine state shared by Generate and streaming.
func (s *Service) buildInferContext(ctx context.Context, req *ollama.GenerateRequest, fourDMode bool) (inference.Context, error) {
	_ = ctx
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		return inference.Context{}, fmt.Errorf("model required")
	}

	entry, ok := s.Registry.Resolve(req.Model)
	forward := s.Infer.Name() == "ollama-forward"
	if !ok && !forward {
		return inference.Context{}, fmt.Errorf("model not found: %q — run: 4dollama pull %s (GGUF into FOURD_MODELS); decoding uses native four_d_engine. Optional: FOURD_INFERENCE=ollama + OLLAMA_HOST for hybrid", req.Model, req.Model)
	}

	var inspectJSON, path string
	if ok {
		path = entry.Path
		if b, err := s.Eng.InspectGGUF(entry.Path); err == nil {
			inspectJSON = string(b)
			if s.Log != nil {
				s.Log.Debug("gguf inspect",
					slog.String("model", req.Model),
					slog.String("path", entry.Path),
					slog.String("inference", s.Infer.Name()),
					slog.String("backend", string(s.Eng.Info().Backend)),
				)
			}
		} else if s.Log != nil {
			s.Log.Warn("gguf inspect failed", slog.String("path", entry.Path), slog.Any("err", err))
		}
	} else if forward && s.Log != nil {
		s.Log.Debug("forwarding without local GGUF resolve", slog.String("model", req.Model))
	}

	demoIn := promptRunesToFloats(req.Prompt)
	if s.Eng != nil && len(demoIn) > 0 {
		if gb := s.Eng.GPUBackend(); gb != "cpu" && s.Log != nil {
			s.Log.Debug("GPU 4D acceleration (CUDA/Metal)", slog.String("gpu", gb))
		}
	}
	var fourDDemo []float32
	if s.Eng != nil && len(demoIn) > 0 {
		var derr error
		fourDDemo, derr = s.Eng.Compute4DDemo(demoIn)
		if derr != nil && s.Log != nil {
			s.Log.Warn("Compute4DDemo", slog.Any("err", derr))
		}
	}
	if len(fourDDemo) > 0 && s.Log != nil {
		s.Log.Debug("4D engine demo (quaternion rotation)",
			slog.Int("demo_in_len", len(demoIn)),
			slog.Int("demo_out_len", len(fourDDemo)))
	}

	var paramCount int64
	var lifted []float32
	var from4DGGUF bool
	ollamaRoot := s.Registry.OllamaDir()
	if ollamaRoot != "" {
		blob4d := models.BlobPath4DGGUF(ollamaRoot, req.Model)
		if w, pc, err := models.Load4DGGUF(blob4d); err == nil && len(w) > 0 {
			lifted = w
			paramCount = pc
			from4DGGUF = true
		}
	}
	if ok && path != "" && s.Eng != nil && len(lifted) == 0 {
		if pc, err := s.Eng.GGUFParamCount(path); err == nil {
			paramCount = pc
		}
		l, p, err := s.Eng.GGUFSampleLift(path, 8192)
		if err == nil && len(l) > 0 {
			lifted = l
			if p > 0 {
				paramCount = p
			}
			if s.Log != nil {
				s.Log.Debug("GGUF sample lifted to 4D",
					slog.String("model", req.Model),
					slog.Int64("param_count", paramCount),
					slog.Int("lift_sample_len", len(lifted)))
			}
			if ollamaRoot != "" {
				blob4d := models.BlobPath4DGGUF(ollamaRoot, req.Model)
				if err := models.Save4DGGUF(blob4d, lifted, paramCount); err != nil && s.Log != nil {
					s.Log.Debug("save .4dgguf", slog.String("path", blob4d), slog.Any("err", err))
				}
			}
		}
	}

	var ropeEmb []float32
	if s.Eng != nil && len(demoIn) > 0 {
		r, rerr := s.Eng.Rope4DSequence(demoIn)
		if rerr != nil && s.Log != nil {
			s.Log.Warn("Rope4DSequence", slog.Any("err", rerr))
		} else {
			ropeEmb = r
		}
	}
	if len(ropeEmb) > 0 && s.Log != nil {
		s.Log.Debug("quaternion RoPE",
			slog.Int("rope_len", len(ropeEmb)),
			slog.Int("rope_quads", len(ropeEmb)/4))
	}

	var attnOut []float32
	if s.Eng != nil && len(ropeEmb) >= 4 {
		seqLen := len(ropeEmb) / 4
		a, aerr := s.Eng.SpacetimeAttention4D(ropeEmb, ropeEmb, ropeEmb, seqLen)
		if aerr != nil && s.Log != nil {
			s.Log.Warn("SpacetimeAttention4D", slog.Any("err", aerr))
		} else if len(a) > 0 {
			attnOut = a
		}
	}
	if len(attnOut) > 0 && s.Log != nil {
		s.Log.Debug("spacetime attention 4D",
			slog.Int("seq_len", len(attnOut)/4),
			slog.Int("attn_floats", len(attnOut)))
	}

	fwd := attnOut
	if len(fwd) == 0 {
		fwd = ropeEmb
	}
	var matmul float32
	gemmOK := false
	if len(fwd) > 0 && len(lifted) > 0 && s.Eng != nil {
		kk := len(fwd)
		if len(lifted) < kk {
			kk = len(lifted)
		}
		row, gerr := s.Eng.Gemm4D(fwd[:kk], lifted[:kk], 1, kk, 1)
		if gerr == nil && len(row) > 0 {
			matmul = row[0]
			gemmOK = true
		} else {
			for i := 0; i < kk; i++ {
				matmul += fwd[i] * lifted[i]
			}
		}
	} else {
		for i := 0; i < len(fwd) && i < len(lifted); i++ {
			matmul += fwd[i] * lifted[i]
		}
	}
	if gemmOK && s.Log != nil {
		s.Log.Debug("native 4D GEMM",
			slog.Bool("from_4dgguf", from4DGGUF),
			slog.Int("lift_len", len(lifted)))
	}

	return inference.Context{
		ModelResolved:      ok,
		ModelPath:          path,
		InspectJSON:        inspectJSON,
		FourDMode:          fourDMode,
		Eng:                s.Eng,
		FourDDemo:          fourDDemo,
		LiftedWeights:      lifted,
		ParamCount:         paramCount,
		RoPEEmbedding:      ropeEmb,
		SpacetimeAttention: attnOut,
		MatmulScore:        matmul,
		WeightsFrom4DGGUF:  from4DGGUF,
	}, nil
}

func stubStreamLog(s *Service, prompt string) {
	if s.Infer.Name() != "stub" || s.Log == nil {
		return
	}
	demoIn := promptRunesToFloats(prompt)
	if len(demoIn) > 0 {
		s.Log.Debug("stub 4D autoregressive path",
			slog.Int("max_tokens", inference.AutoregMaxTokens),
			slog.Int("prompt_embed_len", len(demoIn)))
	}
}

type inferStreamer interface {
	GenerateStream(ctx context.Context, cx inference.Context, model, prompt string, emit func(string) error) error
}

func emitRuneDeltas(text string, emit func(string) error) error {
	for _, r := range text {
		if err := emit(string(r)); err != nil {
			return err
		}
	}
	return nil
}

// Generate produces a completion via the configured inference provider.
func (s *Service) Generate(ctx context.Context, req ollama.GenerateRequest, fourDMode bool) (ollama.GenerateResponse, error) {
	ic, err := s.buildInferContext(ctx, &req, fourDMode)
	if err != nil {
		return ollama.GenerateResponse{}, err
	}
	stubStreamLog(s, req.Prompt)
	text, err := s.Infer.Generate(ctx, ic, req.Model, req.Prompt)
	if err != nil {
		return ollama.GenerateResponse{}, err
	}
	return inference.ToResponse(req.Model, text), nil
}

// StreamGenerate streams completion deltas; native stub and ollama-forward stream from the provider, others fall back to rune-chunking after Generate.
func (s *Service) StreamGenerate(ctx context.Context, req ollama.GenerateRequest, fourDMode bool, emit func(string) error) error {
	ic, err := s.buildInferContext(ctx, &req, fourDMode)
	if err != nil {
		return err
	}
	stubStreamLog(s, req.Prompt)
	if sg, ok := s.Infer.(inferStreamer); ok {
		return sg.GenerateStream(ctx, ic, req.Model, req.Prompt, emit)
	}
	text, err := s.Infer.Generate(ctx, ic, req.Model, req.Prompt)
	if err != nil {
		return err
	}
	return emitRuneDeltas(text, emit)
}

// StreamChat folds chat messages into a prompt and streams the completion.
func (s *Service) StreamChat(ctx context.Context, req ollama.ChatRequest, fourDMode bool, emit func(string) error) error {
	msgs := dedupeConsecutiveUserMessages(ensureFourDSystemPrompt(req.Messages))
	var user strings.Builder
	for _, m := range msgs {
		user.WriteString(m.Role)
		user.WriteString(": ")
		user.WriteString(m.Content)
		user.WriteString("\n")
	}
	gr := ollama.GenerateRequest{Model: req.Model, Prompt: user.String()}
	return s.StreamGenerate(ctx, gr, fourDMode, emit)
}

// Chat maps chat messages to a single prompt and uses Generate.
func (s *Service) Chat(ctx context.Context, req ollama.ChatRequest, fourDMode bool) (ollama.ChatResponse, error) {
	msgs := dedupeConsecutiveUserMessages(ensureFourDSystemPrompt(req.Messages))
	var user strings.Builder
	for _, m := range msgs {
		user.WriteString(m.Role)
		user.WriteString(": ")
		user.WriteString(m.Content)
		user.WriteString("\n")
	}
	g, err := s.Generate(ctx, ollama.GenerateRequest{Model: req.Model, Prompt: user.String()}, fourDMode)
	if err != nil {
		return ollama.ChatResponse{}, err
	}
	return ollama.ChatResponse{
		Model:     g.Model,
		CreatedAt: g.CreatedAt,
		Message:   ollama.Message{Role: "assistant", Content: g.Response},
		Done:      true,
	}, nil
}
