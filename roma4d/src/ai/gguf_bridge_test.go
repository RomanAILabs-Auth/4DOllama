package ai

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseLLMSection_emptyModel(t *testing.T) {
	src := `
[llm]
model_path = ""
n_gpu_layers = 32
`
	c := parseLLMSection(src, "/tmp/pkg")
	if c.CognitiveEnabled {
		t.Fatal("expected cognitive off with empty path")
	}
}

func TestParseLLMSection_withModelFile(t *testing.T) {
	dir := t.TempDir()
	fake := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(fake, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	src := `
[llm]
model_path = "m.gguf"
n_gpu_layers = 16
`
	c := parseLLMSection(src, dir)
	if !c.CognitiveEnabled {
		t.Fatal("expected cognitive on when model file exists")
	}
	if c.ModelPath != fake {
		t.Fatalf("model path %q want %q", c.ModelPath, fake)
	}
	if c.NGPULayers != 16 {
		t.Fatalf("layers %d", c.NGPULayers)
	}
}

func TestExtractPythonSnippetAroundLine(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "x.py")
	body := "a = 1\nfor i in range(3):\n    x = i\n"
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	s := ExtractPythonSnippetAroundLine(p, 2, 2)
	if !strings.Contains(s, "for i in range") {
		t.Fatalf("snippet: %q", s)
	}
}
