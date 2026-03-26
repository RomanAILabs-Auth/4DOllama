package tests

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

func TestMIRToLLVMIRHelloMarkers(t *testing.T) {
	root := roma4dRoot(t)
	ex := filepath.Join(root, "examples", "hello_4d.roma4d")
	res, err := compiler.CheckFile(root, ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatal(res.Errors)
	}
	mir, errs := compiler.LowerToMIR(res)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	ll, warns, err := compiler.MIRToLLVMIR(mir)
	if err != nil {
		t.Log("warnings:", warns)
		t.Fatal(err)
	}
	for _, needle := range []string{
		"define i32 @main",
		"@roma4d.mir.has_soa",
		"@roma4d.mir.has_par",
		"@roma4d.mir.has_unsafe",
		"@roma4d.mir.has_geom",
		"@roma4d.mir.simd_geom",
		"@roma4d.mir.gpu_par_spacetime",
		"@roma4d.mir.has_spacetime_region",
		"@roma4d.mir.has_temporal",
		"@roma4d.mir.has_timetravel_borrow",
		"<4 x double>",
		"fmul",
		"getelementptr",
		"declare",
	} {
		if !strings.Contains(ll, needle) {
			t.Errorf("LLVM IR missing %q", needle)
		}
	}
	// Pass 8: spacetime must not pull in any runtime ledger / epoch / chrono symbols.
	for _, forbidden := range []string{
		"roma4d_chrono",
		"roma4d_advance_epoch",
		"roma4d_current_epoch",
	} {
		if strings.Contains(ll, forbidden) {
			t.Errorf("LLVM IR must not contain runtime spacetime helper %q (compile-time only)", forbidden)
		}
	}
	t.Log(ll[:min(800, len(ll))], "...")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestBuildExecutableMinMain(t *testing.T) {
	if compiler.FindClang() == "" {
		t.Skip("clang not on PATH")
	}
	root := roma4dRoot(t)
	src := filepath.Join(root, "examples", "min_main.roma4d")
	out := filepath.Join(t.TempDir(), "min_main_out")
	if runtime.GOOS == "windows" {
		out += ".exe"
	}
	warns, err := compiler.BuildExecutable(root, src, out, nil)
	if len(warns) > 0 {
		t.Log("warnings:", warns)
	}
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(out)
	err = cmd.Run()
	var code int
	if err != nil {
		if x, ok := err.(*exec.ExitError); ok {
			code = x.ExitCode()
		} else {
			t.Fatal(err)
		}
	}
	if code != 42 {
		t.Fatalf("expected exit code 42, got %d", code)
	}
}
