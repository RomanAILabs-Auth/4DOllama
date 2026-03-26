package compiler

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveRoma4DModuleFile_r4sPreferred(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "roma4d.toml"), []byte("name = \"t\"\nversion=\"0\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	libR4s := filepath.Join(dir, "libgeo.r4s")
	if err := os.WriteFile(libR4s, []byte("def f() -> int:\n    return 0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := ResolveRoma4DModuleFile(dir, "libgeo")
	if got != libR4s {
		t.Fatalf("want %s got %s", libR4s, got)
	}
}

func TestResolveRoma4DModuleFile_legacyRoma4d(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "roma4d.toml"), []byte("name = \"t\"\nversion=\"0\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	legacy := filepath.Join(dir, "legacy.roma4d")
	if err := os.WriteFile(legacy, []byte("def f() -> int:\n    return 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := ResolveRoma4DModuleFile(dir, "legacy")
	if got != legacy {
		t.Fatalf("want %s got %s", legacy, got)
	}
}

func TestIsRoma4DSourcePath(t *testing.T) {
	if !IsRoma4DSourcePath("x.r4s") || !IsRoma4DSourcePath("x.roma4d") {
		t.Fatal()
	}
	if IsRoma4DSourcePath("x.py") {
		t.Fatal()
	}
}
