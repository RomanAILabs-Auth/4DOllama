package tests

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
	"github.com/RomanAILabs-Auth/Roma4D/src/parser"
)

func roma4dRoot(t *testing.T) string {
	t.Helper()
	return filepath.Join("..")
}

func TestCheckHelloExample(t *testing.T) {
	root := roma4dRoot(t)
	ex := filepath.Join(root, "examples", "hello_4d.r4d")
	res, err := compiler.CheckFile(root, ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("typecheck errors: %v", res.Errors)
	}
	if res.Module.Qual == "" {
		t.Fatal("expected module qual set")
	}
}

func TestCheckBench4dExample(t *testing.T) {
	root := roma4dRoot(t)
	ex := filepath.Join(root, "examples", "Bench_4d.r4d")
	res, err := compiler.CheckFile(root, ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("Bench_4d.r4d typecheck errors: %v", res.Errors)
	}
}

func TestNameResolutionAndGAOps(t *testing.T) {
	src := `
def f():
    v = vec4(x=1, y=0, z=0, w=0)
    r = rotor(angle=1.0, plane="xy")
    a = v * r
    b = v ^ v
    c = v | v
    i = 1 ^ 2
    return a
`
	dir := t.TempDir()
	man := filepath.Join(dir, "roma4d.toml")
	if err := os.WriteFile(man, []byte("name = \"tmp\"\nversion = \"0.0.1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tmp := filepath.Join(dir, "x.r4d")
	if err := os.WriteFile(tmp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := compiler.CheckFile(dir, tmp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("unexpected: %v", res.Errors)
	}
}

func TestImportErrorMessage(t *testing.T) {
	src := `import does_not_exist_roma4d_module`
	dir := t.TempDir()
	man := filepath.Join(dir, "roma4d.toml")
	if err := os.WriteFile(man, []byte("name = \"tmp\"\nversion = \"0.0.1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	tmp := filepath.Join(dir, "x.r4d")
	if err := os.WriteFile(tmp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := compiler.CheckFile(dir, tmp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) == 0 {
		t.Fatal("expected ImportError")
	}
}

func TestParseStillWorks(t *testing.T) {
	_, err := parser.Parse("x.roma4d", "x = 1\n")
	if err != nil {
		t.Fatal(err)
	}
}
