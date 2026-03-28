package runner

import (
	"context"
	"log/slog"
	"strings"

	"github.com/4dollama/4dollama/internal/ollama"
)

// romanAIChatPrompt is the default system instruction for /api/chat: natural RomanAI persona, no internal stack jargon.
const romanAIChatPrompt = `You are RomanAI, a helpful assistant from RomanAILabs. Reply in clear, natural, friendly English. Match the user's tone. Be concise unless they ask for more detail. Do not mention implementation details, environment variables, or engine internals unless the user explicitly asks how the technology works.`

// romanAITechnicalAppendix is merged into the system prompt only when debug logging is enabled (e.g. serve -verbose).
const romanAITechnicalAppendix = `

[Operator/debug context — only discuss if the user asks about the stack:]
4dollama can route completions through a native geometric engine and/or upstream Ollama depending on configuration. Cite docs or operator settings rather than inventing APIs.`

// RomanAIFriendlyFallback is returned when the stub path cannot produce human-quality text (e.g. .4dai without a working upstream).
const RomanAIFriendlyFallback = "I'm here and ready to chat, but I couldn't finish a proper reply just yet. Please start Ollama with this same model name available (create it there from your weights if needed), then try again—I’ll answer in clear, natural English."

func ensureChatSystemPrompt(ctx context.Context, msgs []ollama.Message, log *slog.Logger) []ollama.Message {
	if ctx == nil {
		ctx = context.Background()
	}
	base := romanAIChatPrompt
	if log != nil && log.Enabled(ctx, slog.LevelDebug) {
		base = romanAIChatPrompt + romanAITechnicalAppendix
	}
	if len(msgs) == 0 {
		return []ollama.Message{{Role: "system", Content: base}}
	}
	if strings.EqualFold(strings.TrimSpace(msgs[0].Role), "system") {
		first := msgs[0]
		merged := base
		if strings.TrimSpace(first.Content) != "" {
			merged = base + "\n\n" + first.Content
		}
		out := make([]ollama.Message, 0, len(msgs))
		out = append(out, ollama.Message{Role: "system", Content: merged})
		out = append(out, msgs[1:]...)
		return out
	}
	out := make([]ollama.Message, 0, len(msgs)+1)
	out = append(out, ollama.Message{Role: "system", Content: base})
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
