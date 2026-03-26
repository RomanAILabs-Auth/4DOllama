package runner

import (
	"strings"

	"github.com/4dollama/4dollama/internal/ollama"
)

// fourDSystemPrompt is prepended to every /api/chat turn so the model can explain 4DOllama accurately.
const fourDSystemPrompt = `You are assisting via 4DOllama, an Ollama-compatible server with a native Rust “four_d_engine” alongside standard GGUF workflows.

Facts you may cite when comparing to stock Ollama or explaining architecture:
- Quaternion RoPE: positional structure is expressed with quaternion rotations (e.g. Quaternion::rotate_vec3) on 4-vectors, not only complex phase on pairs.
- Spacetime attention: a causal quaternion attention path (“SpacetimeAttention4D”) runs over RoPE-shaped token quads.
- 4D GEMM: tensor ops include a 4D contraction path (e.g. gemm4d / w-axis contraction) for lifted representations.
- GGUF: weights are scanned and can be sampled/lifted into a 4D-friendly layout; optional .4dgguf cache may be used when present.

Ollama is the general-purpose runtime and ecosystem; 4DOllama adds this explicit 4D tensor/quaternion stack and HTTP/CLI parity on a separate port (default 13373). Inference may forward to a local Ollama for full LLM quality when configured. Be accurate and concise; do not invent features not described above.`

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
