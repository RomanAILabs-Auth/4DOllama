package cli

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// CmdPush attempts upstream registry push via the stock `ollama` CLI when present (4dollama cloud TBD).
func CmdPush(args []string, log *slog.Logger) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: 4dollama push <model>[:tag]")
		return 2
	}
	if path, err := exec.LookPath("ollama"); err == nil && path != "" {
		cmd := exec.Command("ollama", append([]string{"push"}, args...)...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			if log != nil {
				log.Warn("ollama push delegate failed", slog.Any("err", err))
			}
			return 1
		}
		return 0
	}
	fmt.Fprintln(os.Stderr, "4dollama push: install `ollama` on PATH to push to registry.ollama.ai, or use the web UI.")
	fmt.Fprintln(os.Stderr, "Local 4D models live under FOURD_MODELS (~/.4dollama/models) including .4dai RomanAI carriers.")
	return 1
}

// CmdSignin delegates to `ollama signin` when available.
func CmdSignin(args []string, log *slog.Logger) int {
	if _, err := exec.LookPath("ollama"); err == nil {
		cmd := exec.Command("ollama", append([]string{"signin"}, args...)...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintln(os.Stderr, "4dollama signin: `ollama` not found on PATH.")
	fmt.Fprintln(os.Stderr, "Sign in at https://ollama.com and install the Ollama app, or set OLLAMA_HOST for registry pulls.")
	_ = log
	return 1
}

// CmdSignout delegates to `ollama signout` when available.
func CmdSignout(args []string, log *slog.Logger) int {
	if _, err := exec.LookPath("ollama"); err == nil {
		cmd := exec.Command("ollama", append([]string{"signout"}, args...)...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintln(os.Stderr, "4dollama signout: `ollama` not found on PATH (nothing to clear in 4dollama).")
	_ = log
	return 0
}

// CmdLaunch opens the Ollama desktop app / site integration (best-effort).
func CmdLaunch(args []string, log *slog.Logger) int {
	target := strings.TrimSpace(strings.Join(args, " "))
	if target == "" {
		target = "https://ollama.com"
	}
	switch runtime.GOOS {
	case "windows":
		_ = exec.Command("cmd", "/c", "start", "", target).Start()
	case "darwin":
		_ = exec.Command("open", target).Start()
	default:
		_ = exec.Command("xdg-open", target).Start()
	}
	if log != nil {
		log.Info("launch", slog.String("target", target))
	}
	fmt.Println("launch:", target)
	return 0
}
