package compiler

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// BuildDebugToStderr is true when R4D_DEBUG is 1/true/yes — failure logs are also printed to stderr.
func BuildDebugToStderr() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("R4D_DEBUG")))
	return v == "1" || v == "true" || v == "yes"
}

// WriteBuildFailureLog appends a diagnostic block to pkgRoot/debug/last_build_failure.log.
func WriteBuildFailureLog(pkgRoot, stage string, lines [][2]string) {
	if pkgRoot == "" {
		return
	}
	dir := filepath.Join(pkgRoot, "debug")
	_ = os.MkdirAll(dir, 0o755)
	p := filepath.Join(dir, "last_build_failure.log")

	var b strings.Builder
	b.WriteString(fmt.Sprintf("=== Roma4D build failure %s ===\n", time.Now().UTC().Format(time.RFC3339)))
	b.WriteString(fmt.Sprintf("stage: %s\n", stage))
	b.WriteString(fmt.Sprintf("host_go: %s/%s\n", runtime.GOOS, runtime.GOARCH))
	for _, kv := range lines {
		if len(kv) != 2 {
			continue
		}
		b.WriteString(fmt.Sprintf("--- %s ---\n%s\n\n", kv[0], strings.TrimSpace(kv[1])))
	}
	b.WriteByte('\n')

	f, err := os.OpenFile(p, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	_, _ = f.WriteString(b.String())
	_ = f.Close()

	if BuildDebugToStderr() {
		_, _ = fmt.Fprint(os.Stderr, b.String())
	}
}

func readFileHead(path string, max int) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("(read error: %v)", err)
	}
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + fmt.Sprintf("\n... truncated (%d bytes total)", len(b))
}

func formatArgv0(name string, args []string) string {
	var b strings.Builder
	b.WriteString(name)
	for _, a := range args {
		b.WriteByte(' ')
		if strings.ContainsAny(a, " \t\"") {
			b.WriteString(fmt.Sprintf("%q", a))
		} else {
			b.WriteString(a)
		}
	}
	return b.String()
}
