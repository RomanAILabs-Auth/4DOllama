package ollama

// JSON shapes aligned with Ollama HTTP API (subset used by 4dollama).

type GenerateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  *bool          `json:"stream,omitempty"`
	Options map[string]any `json:"options,omitempty"`
}

type GenerateResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
	Done      bool   `json:"done"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string         `json:"model"`
	Messages []Message      `json:"messages"`
	Stream   *bool          `json:"stream,omitempty"`
	Options  map[string]any `json:"options,omitempty"`
}

type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt string    `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`
}

type Model struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Size       int64  `json:"size"`
	Digest     string `json:"digest"`
	Details    any    `json:"details"`
}

type TagsResponse struct {
	Models []Model `json:"models"`
}

type VersionResponse struct {
	Version string `json:"version"`
}

type PullRequest struct {
	Name   string `json:"name"`
	Stream *bool  `json:"stream,omitempty"`
}

type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

type PsResponse struct {
	Models []any `json:"models"`
}

type EmbeddingsRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingsResponse struct {
	Embedding []float32 `json:"embedding"`
}

// OpenAI-compatible (minimal).

type OAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OAIChatRequest struct {
	Model    string           `json:"model"`
	Messages []OAIChatMessage `json:"messages"`
	Stream   *bool            `json:"stream,omitempty"`
}

type OAIChatChoice struct {
	Index        int            `json:"index"`
	Message      OAIChatMessage `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type OAIChatResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []OAIChatChoice `json:"choices"`
}

type OAICompletionRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream *bool  `json:"stream,omitempty"`
}

type OAICompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type OAICompletionResponse struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []OAICompletionChoice `json:"choices"`
}
