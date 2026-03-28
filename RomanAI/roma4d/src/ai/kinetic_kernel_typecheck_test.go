package ai

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

func TestKineticSampleKernelTypechecks(t *testing.T) {
	root, err := FindRoma4dPackageRoot(".")
	if err != nil {
		t.Skip(err)
	}
	L := LiftLoop{
		Liftable:   true,
		Lineno:     1,
		Target:     "k_i",
		AccVar:     "k_acc",
		RangeStop:  "1000",
		RangeStart: "0",
		RangeStep:  "1",
		R4dRhsLine: "k_acc = k_acc + (float(k_i) * float(k_i))",
	}
	src, err := buildKineticModuleSource(L)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "kinetic_sample.r4d")
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	cr, err := compiler.CheckFile(root, p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(cr.Errors) > 0 {
		t.Fatal(cr.Errors)
	}
}
