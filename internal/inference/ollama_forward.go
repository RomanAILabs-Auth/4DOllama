package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/4dollama/4dollama/internal/ollama"
)

// OllamaForward proxies completion to a real Ollama server (same HTTP API).
type OllamaForward struct {
	BaseURL string
	Client  *http.Client
}

func (OllamaForward) Name() string { return "ollama-forward" }

func (o OllamaForward) client() *http.Client {
	if o.Client != nil {
		return o.Client
	}
	return &http.Client{Timeout: 0}
}

// Generate calls upstream /api/generate (non-streaming).
func (o OllamaForward) Generate(ctx context.Context, _ Context, model, prompt string) (string, error) {
	stream := false
	body, err := json.Marshal(ollama.GenerateRequest{
		Model: model, Prompt: prompt, Stream: &stream,
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client().Do(req)
	if err != nil {
		return "", fmt.Errorf("upstream ollama: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("upstream ollama HTTP %d: %s", resp.StatusCode, string(bytes.TrimSpace(raw)))
	}
	var out ollama.GenerateResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("upstream ollama json: %w", err)
	}
	return out.Response, nil
}
