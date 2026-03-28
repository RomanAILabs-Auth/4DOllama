package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadTokenizerFromModelfile(t *testing.T) {
	dir := t.TempDir()
	mf := filepath.Join(dir, "m.Modelfile")
	content := "FROM ./w.4dai\nTOKENIZER_FROM C:\\\\models\\\\tok.gguf\n"
	if err := os.WriteFile(mf, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	got := ReadTokenizerFromModelfile(mf)
	want := `C:\models\tok.gguf`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
