package inference

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

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

// GenerateStream proxies upstream /api/generate with stream=true and forwards each response delta.
func (o OllamaForward) GenerateStream(ctx context.Context, _ Context, model, prompt string, emit func(string) error) error {
	if emit == nil {
		return nil
	}
	stream := true
	body, err := json.Marshal(ollama.GenerateRequest{
		Model: model, Prompt: prompt, Stream: &stream,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client().Do(req)
	if err != nil {
		return fmt.Errorf("upstream ollama: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upstream ollama HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	sc := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, 4*1024*1024)
	for sc.Scan() {
		var ev struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		if ev.Response != "" {
			if err := emit(ev.Response); err != nil {
				return err
			}
		}
		if ev.Done {
			break
		}
	}
	return sc.Err()
}

// Chat calls upstream /api/chat (non-streaming).
func (o OllamaForward) Chat(ctx context.Context, model string, messages []ollama.Message) (ollama.ChatResponse, error) {
	stream := false
	body, err := json.Marshal(ollama.ChatRequest{
		Model: model, Messages: messages, Stream: &stream,
	})
	if err != nil {
		return ollama.ChatResponse{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return ollama.ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client().Do(req)
	if err != nil {
		return ollama.ChatResponse{}, fmt.Errorf("upstream ollama: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ollama.ChatResponse{}, err
	}
	if resp.StatusCode >= 400 {
		return ollama.ChatResponse{}, fmt.Errorf("upstream ollama HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out ollama.ChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return ollama.ChatResponse{}, fmt.Errorf("upstream ollama json: %w", err)
	}
	return out, nil
}

// ChatStream proxies upstream /api/chat with stream=true and forwards message content deltas.
func (o OllamaForward) ChatStream(ctx context.Context, model string, messages []ollama.Message, emit func(string) error) error {
	if emit == nil {
		return nil
	}
	stream := true
	body, err := json.Marshal(ollama.ChatRequest{
		Model: model, Messages: messages, Stream: &stream,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client().Do(req)
	if err != nil {
		return fmt.Errorf("upstream ollama: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upstream ollama HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	sc := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, 4*1024*1024)
	for sc.Scan() {
		var ev struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		if ev.Message.Content != "" {
			if err := emit(ev.Message.Content); err != nil {
				return err
			}
		}
		if ev.Done {
			break
		}
	}
	return sc.Err()
}
