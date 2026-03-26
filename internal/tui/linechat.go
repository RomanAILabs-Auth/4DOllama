package tui

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/4dollama/4dollama/internal/ollama"
)

// RunLineChat is Ollama-identical line REPL: only ">>> " prompts, no header/footer (Windows ConPTY-safe).
func RunLineChat(modelName, base string) error {
	base = strings.TrimSuffix(base, "/")
	messages := make([]ollama.Message, 0, 32)
	in := bufio.NewReader(os.Stdin)
	for {
		fmt.Fprint(os.Stdout, ">>> ")
		_ = os.Stdout.Sync()
		line, err := in.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		text := strings.TrimSpace(line)
		if text == "" {
			continue
		}
		low := strings.ToLower(text)
		if low == "/bye" || low == "/exit" || low == "/quit" {
			return nil
		}
		if low == "/help" || low == "/?" {
			fmt.Println("Available Commands:")
			fmt.Println("  /clear          Clear the session context")
			fmt.Println("  /bye            Exit")
			continue
		}
		if low == "/clear" {
			messages = messages[:0]
			continue
		}
		messages = append(messages, ollama.Message{Role: "user", Content: text})
		sw := &StreamingPlainWriter{Out: os.Stdout}
		assistant, err := streamChatRound(sw, base, modelName, messages)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
			messages = messages[:len(messages)-1]
			continue
		}
		fmt.Fprintln(os.Stdout)
		if assistant != "" {
			messages = append(messages, ollama.Message{Role: "assistant", Content: assistant})
		}
	}
}

func streamChatRound(sw *StreamingPlainWriter, base, model string, messages []ollama.Message) (string, error) {
	if sw != nil {
		sw.Reset()
	}
	tr := true
	body := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   tr,
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest(http.MethodPost, base+"/api/chat", bytes.NewReader(buf))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("%w (is `4dollama serve` running on %s?)", err, base)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	var full strings.Builder
	sc := bufio.NewScanner(resp.Body)
	buf2 := make([]byte, 0, 64*1024)
	sc.Buffer(buf2, 4*1024*1024)
	for sc.Scan() {
		var ev struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			Done bool `json:"done"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		if ev.Message.Content != "" {
			full.WriteString(ev.Message.Content)
			if sw != nil {
				sw.WriteChunk(ev.Message.Content)
			}
			_ = os.Stdout.Sync()
		}
		if ev.Done {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return "", err
	}
	return full.String(), nil
}
