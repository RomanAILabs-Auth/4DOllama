package tests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

func TestLowerHelloExampleToMIR(t *testing.T) {
	root := roma4dRoot(t)
	ex := filepath.Join(root, "examples", "hello_4d.roma4d")
	res, err := compiler.CheckFile(root, ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("check errors: %v", res.Errors)
	}
	mir, errs := compiler.LowerToMIR(res)
	if len(errs) > 0 {
		t.Fatalf("LowerToMIR: %v", errs)
	}
	if mir == nil {
		t.Fatal("nil MIR")
	}
	out := mir.String()
	t.Log(out)
	for _, needle := range []string{
		"module",
		"fn main",
		"par_region",
		"unsafe_region",
		"soa_load",
		"heap_alloc",
		"ptr_store",
		"ptr_load",
		"class Particle",
	} {
		if !strings.Contains(out, needle) {
			t.Errorf("MIR dump missing %q", needle)
		}
	}
}

func TestLowerToMIRRejectsCheckErrors(t *testing.T) {
	dir := tmpPkg(t)
	src := `def f():
    x = 1
    x = y_undefined_zzz
`
	p := filepath.Join(dir, "bad.roma4d")
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := compiler.CheckFile(dir, p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) == 0 {
		t.Fatal("expected type errors")
	}
	mir, errs := compiler.LowerToMIR(res)
	if mir != nil {
		t.Fatal("expected nil MIR")
	}
	if len(errs) == 0 {
		t.Fatal("expected diagnostics")
	}
}
