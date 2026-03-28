// romanai — RomanAI host launcher: always runs kernel with this checkout's r4d (go run ./cmd/r4d).
// Install: go install ./cmd/romanai   →  %GOPATH%\bin\romanai.exe
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	os.Exit(run())
}

func run() int {
	if len(os.Args) < 2 {
		usage()
		return 1
	}
	switch os.Args[1] {
	case "-h", "--help", "help":
		usage()
		return 0
	case "run":
		return cmdRun(os.Args[2:])
	case "chat", "version", "list":
		return cmdKernelOnly(os.Args[1])
	default:
		fmt.Fprintf(os.Stderr, "romanai: unknown command %q\n\n", os.Args[1])
		usage()
		return 1
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `RomanAI launcher (uses bundled roma4d via go run; avoids stale PATH r4d).

  romanai run  path\to\model.gguf [prompt words...]
  romanai chat
  romanai version
  romanai list

If not inside the repo, set ROMANAI_ROOT to your RomanAI folder
(e.g. C:\Users\You\Desktop\4DEngine\RomanAI).

Or copy romanai.exe into that RomanAI folder (next to romanai.cmd).

`)
}

func findRomanAIRoot() (string, error) {
	if e := strings.TrimSpace(os.Getenv("ROMANAI_ROOT")); e != "" {
		if ok, err := isRomanAIRoot(e); ok {
			return filepath.Clean(e), nil
		} else if err != nil {
			return "", fmt.Errorf("ROMANAI_ROOT %q: %w", e, err)
		}
		return "", fmt.Errorf("ROMANAI_ROOT %q is not a RomanAI checkout (need r4d\\romanai_main.r4d and roma4d\\go.mod)", e)
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		if ok, _ := isRomanAIRoot(exeDir); ok {
			return filepath.Clean(exeDir), nil
		}
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for d := filepath.Clean(wd); d != ""; {
		if ok, _ := isRomanAIRoot(d); ok {
			return d, nil
		}
		cand := filepath.Join(d, "RomanAI")
		if ok, _ := isRomanAIRoot(cand); ok {
			return filepath.Clean(cand), nil
		}
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}
	return "", fmt.Errorf(`could not find RomanAI root (expected r4d\romanai_main.r4d and roma4d\go.mod).
Set environment variable ROMANAI_ROOT to your RomanAI folder, or cd into that repo.`)
}

func isRomanAIRoot(dir string) (bool, error) {
	kernel := filepath.Join(dir, "r4d", "romanai_main.r4d")
	gomod := filepath.Join(dir, "roma4d", "go.mod")
	st1, e1 := os.Stat(kernel)
	st2, e2 := os.Stat(gomod)
	if e1 != nil || e2 != nil {
		return false, nil
	}
	if st1.IsDir() || st2.IsDir() {
		return false, nil
	}
	return true, nil
}

func cmdRun(args []string) int {
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "usage: romanai run path\\to\\model.gguf [prompt...]\n")
		return 1
	}
	modelArg := args[0]
	promptText := "run"
	if len(args) >= 2 {
		promptText = strings.Join(args[1:], " ")
	}
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	full := modelArg
	if !filepath.IsAbs(full) {
		full = filepath.Join(wd, full)
	}
	full, err = filepath.Abs(full)
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	st, err := os.Stat(full)
	if err != nil || st.IsDir() {
		fmt.Fprintf(os.Stderr, "romanai run: GGUF not found: %s\n", full)
		return 1
	}
	ext := strings.ToLower(filepath.Ext(full))
	if ext != ".gguf" {
		fmt.Fprintf(os.Stderr, "romanai: warning: expected .gguf extension (got %s)\n", ext)
	}

	root, err := findRomanAIRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	roma4d := filepath.Join(root, "roma4d")
	kernel := filepath.Join(root, "r4d", "romanai_main.r4d")
	kAbs, err := filepath.Abs(kernel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}

	_ = os.Setenv("ROMANAI_CLI_MODEL", full)
	_ = os.Setenv("ROMANAI_GGUF", full)
	_ = os.Setenv("ROMANAI_PROMPT", promptText)
	_ = os.Setenv("R4D_EXPERT_INTERACTIVE", "0")

	cmd := exec.Command("go", "run", "./cmd/r4d", "run", kAbs)
	cmd.Dir = roma4d
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	return 0
}

func cmdKernelOnly(prompt string) int {
	root, err := findRomanAIRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	roma4d := filepath.Join(root, "roma4d")
	kernel := filepath.Join(root, "r4d", "romanai_main.r4d")
	kAbs, err := filepath.Abs(kernel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	_ = os.Setenv("ROMANAI_PROMPT", prompt)
	_ = os.Setenv("R4D_EXPERT_INTERACTIVE", "0")

	cmd := exec.Command("go", "run", "./cmd/r4d", "run", kAbs)
	cmd.Dir = roma4d
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "romanai: %v\n", err)
		return 1
	}
	return 0
}
