package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

// CmdStop calls POST /api/stop (Ollama-compatible surface). No in-process runners yet — always succeeds.
func CmdStop(model string, log *slog.Logger) int {
	model = strings.TrimSpace(model)
	if model == "" {
		fmt.Fprintln(os.Stderr, "usage: 4dollama stop <model>")
		return 2
	}
	if i := strings.IndexByte(model, ':'); i >= 0 {
		model = model[:i]
	}
	body, _ := json.Marshal(map[string]string{"name": model})
	req, err := http.NewRequest(http.MethodPost, baseURL()+"/api/stop", bytes.NewReader(body))
	if err != nil {
		return 1
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama stop: %v (is the server running? try `4dollama serve`)\n", err)
		return 1
	}
	defer resp.Body.Close()
	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		fmt.Fprintf(os.Stderr, "4dollama stop: %s\n", strings.TrimSpace(string(bodyBytes)))
		return 1
	}
	if log != nil {
		log.Info("stop acknowledged", slog.String("model", model))
	}
	if len(bytes.TrimSpace(bodyBytes)) > 0 {
		fmt.Println(string(bodyBytes))
	} else {
		fmt.Printf("stopped %s\n", model)
	}
	return 0
}
