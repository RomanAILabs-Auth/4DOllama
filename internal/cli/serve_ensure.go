package cli

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

func mergeEnv(base []string, key, val string) []string {
	prefix := key + "="
	out := make([]string, 0, len(base)+1)
	for _, e := range base {
		if strings.HasPrefix(e, prefix) {
			continue
		}
		out = append(out, e)
	}
	return append(out, prefix+val)
}

// ensureServerRunning starts `4dollama serve` from the same binary if /healthz is not reachable.
// silentChild uses FOURD_LOG_LEVEL=error for the child so interactive chat stays clean.
func ensureServerRunning(log *slog.Logger, silentChild bool) error {
	base := baseURL()
	client := &http.Client{Timeout: 2 * time.Second}
	if resp, err := client.Get(base + "/healthz"); err == nil {
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return nil
		}
	}
	exe, err := os.Executable()
	if err != nil {
		return fmt.Errorf("executable path: %w", err)
	}
	cmd := exec.Command(exe, "serve")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if silentChild {
		cmd.Env = mergeEnv(os.Environ(), "FOURD_LOG_LEVEL", "error")
	} else {
		cmd.Env = os.Environ()
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start serve: %w", err)
	}
	if log != nil && !silentChild {
		log.Info("started local 4dollama serve", slog.Int("pid", cmd.Process.Pid), slog.String("url", base))
	}
	for i := 0; i < 100; i++ {
		time.Sleep(100 * time.Millisecond)
		resp, err := client.Get(base + "/healthz")
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
	}
	return fmt.Errorf("server did not become ready at %s (check logs)", base)
}
