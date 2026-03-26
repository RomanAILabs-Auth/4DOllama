package runner

import (
	"strings"

	"github.com/4dollama/4dollama/internal/ollama"
)

// fourDSystemPrompt is prepended to every /api/chat turn so the model can explain 4DOllama accurately.
const fourDSystemPrompt = `You are assisting via 4DOllama: Ollama-compatible HTTP/CLI, but completions run through the native Rust four_d_engine 4D stack on GGUF you pulled into this server—not through stock llama.cpp unless the operator explicitly enabled hybrid mode.

Architecture to cite accurately:
- Quaternion RoPE on 4-vectors (e.g. Quaternion::rotate_vec3), not only complex phase on pairs.
- SpacetimeAttention4D: causal quaternion attention over RoPE-shaped token quads.
- 4D GEMM / w-axis contraction on lifted tensor views.
- GGUF: manifest scan, weight sample/lift, optional .4dgguf cache; autoregressive decode samples vocab via ProjectStubLogits + four_d_engine.

Default inference is native 4D decode. Optional hybrid (FOURD_INFERENCE=ollama + OLLAMA_HOST) is off unless configured. Be concise; do not invent features.`

func ensureFourDSystemPrompt(msgs []ollama.Message) []ollama.Message {
	if len(msgs) == 0 {
		return []ollama.Message{{Role: "system", Content: fourDSystemPrompt}}
	}
	if strings.EqualFold(strings.TrimSpace(msgs[0].Role), "system") {
		first := msgs[0]
		merged := fourDSystemPrompt
		if strings.TrimSpace(first.Content) != "" {
			merged = fourDSystemPrompt + "\n\n" + first.Content
		}
		out := make([]ollama.Message, 0, len(msgs))
		out = append(out, ollama.Message{Role: "system", Content: merged})
		out = append(out, msgs[1:]...)
		return out
	}
	out := make([]ollama.Message, 0, len(msgs)+1)
	out = append(out, ollama.Message{Role: "system", Content: fourDSystemPrompt})
	out = append(out, msgs...)
	return out
}

func dedupeConsecutiveUserMessages(msgs []ollama.Message) []ollama.Message {
	if len(msgs) == 0 {
		return msgs
	}
	out := []ollama.Message{msgs[0]}
	for i := 1; i < len(msgs); i++ {
		cur := msgs[i]
		last := out[len(out)-1]
		if strings.EqualFold(strings.TrimSpace(last.Role), "user") &&
			strings.EqualFold(strings.TrimSpace(cur.Role), "user") &&
			strings.TrimSpace(last.Content) == strings.TrimSpace(cur.Content) {
			continue
		}
		out = append(out, cur)
	}
	return out
}
