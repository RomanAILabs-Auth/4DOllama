package cli

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"time"
)

// ensureServerRunning starts `4dollama serve` from the same binary if /healthz is not reachable.
func ensureServerRunning(log *slog.Logger) error {
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
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start serve: %w", err)
	}
	if log != nil {
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
