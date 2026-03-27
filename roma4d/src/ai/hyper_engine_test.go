package ai

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestAnalyzePython_SimpleLoop(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "x.py")
	src := `
import math
for i in range(100):
    x = i * 2
`
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	exe, prefix, err := findTestPython(t)
	if err != nil {
		t.Skip(err)
	}
	plan, err := AnalyzePython(context.Background(), p, exe, prefix, DefaultHyperOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !plan.SyntaxOK {
		t.Fatalf("syntax: %s", plan.SyntaxErrorMsg)
	}
	if plan.ASTSummary.ForLoops < 1 {
		t.Fatalf("expected for_loops >= 1, got %+v", plan.ASTSummary)
	}
	found := false
	for _, f := range plan.Findings {
		if f.Category == "parallel_loop_candidate" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected parallel_loop_candidate finding, got %#v", plan.Findings)
	}
}

func findTestPython(t *testing.T) (exe string, prefix []string, err error) {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		if p, e := exec.LookPath(name); e == nil {
			return p, nil, nil
		}
	}
	if runtime.GOOS == "windows" {
		if p, e := exec.LookPath("py"); e == nil {
			return p, []string{"-3"}, nil
		}
	}
	return "", nil, errors.New("python not on PATH")
}
