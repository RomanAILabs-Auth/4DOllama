package compiler

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// MIRToLLVMIR lowers MIR to textual LLVM IR (for tests and debugging).
func MIRToLLVMIR(m *MIRModule) (irText string, warnings []string, err error) {
	mod, w, err := LowerMIRToLLVM(m)
	if err != nil {
		return "", w, err
	}
	s, err2 := LLVMModuleString(mod)
	if err2 != nil {
		return "", w, err2
	}
	return s, w, nil
}

// FindClang returns the first clang executable on PATH, or "".
func FindClang() string {
	for _, name := range []string{"clang", "clang-18", "clang-17", "clang-16", "clang.exe"} {
		p, err := exec.LookPath(name)
		if err == nil {
			return p
		}
	}
	return ""
}

// windowsClangTarget returns -target for clang on Windows so we link against MinGW (GNU) CRT
// instead of the MSVC default (lld-link + libcmt.lib), which requires Visual Studio.
// Install a MinGW-w64 toolchain (e.g. MSYS2) and ensure gcc / mingw bin is on PATH.
func windowsClangTarget() []string {
	if runtime.GOOS != "windows" {
		return nil
	}
	switch runtime.GOARCH {
	case "arm64":
		return []string{"-target", "aarch64-pc-windows-gnu"}
	case "386":
		return []string{"-target", "i686-pc-windows-gnu"}
	default:
		return []string{"-target", "x86_64-pc-windows-gnu"}
	}
}

// windowsGNUChain returns extra clang flags so standalone LLVM Clang finds MinGW headers
// (e.g. mm_malloc.h next to MSYS2 ucrt64/include/malloc.h). Without this, PATH may pull in
// MSYS2 libc headers while Clang never adds GCC's internal include directory.
//
// Override: set R4D_GNU_ROOT to the MinGW prefix (directory that contains bin/ and lib/).
func windowsGNUChain() []string {
	if runtime.GOOS != "windows" {
		return nil
	}
	candidates := []string{}
	if p := strings.TrimSpace(os.Getenv("R4D_GNU_ROOT")); p != "" {
		candidates = append(candidates, filepath.Clean(p))
	}
	for _, name := range []string{"gcc", "gcc.exe", "x86_64-w64-mingw32-gcc.exe"} {
		gccPath, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		binDir := filepath.Dir(gccPath)
		candidates = append(candidates, filepath.Dir(binDir))
	}
	// Typical MSYS2 installs (PowerShell often lacks MINGW_PREFIX).
	for _, p := range []string{
		`C:\msys64\ucrt64`,
		`C:\msys64\mingw64`,
		`C:\msys64\clang64`,
	} {
		candidates = append(candidates, p)
	}
	seen := map[string]struct{}{}
	for _, root := range candidates {
		if root == "" {
			continue
		}
		if _, ok := seen[root]; ok {
			continue
		}
		seen[root] = struct{}{}
		libGCC := filepath.Join(root, "lib", "gcc")
		if st, err := os.Stat(libGCC); err != nil || !st.IsDir() {
			continue
		}
		// Clang wants forward slashes on Windows.
		root = strings.ReplaceAll(filepath.Clean(root), `\`, `/`)
		return []string{"--gcc-toolchain=" + root}
	}
	return nil
}

// BuildExecutable parses and checks roma4dPath, lowers to MIR → LLVM IR, then invokes clang
// to produce a native executable at outExe (add .exe on Windows if needed).
// If rtC exists at pkgRoot/rt/roma4d_rt.c it is linked for unresolved Roma4D runtime symbols.
//
// On failure, details are appended to pkgRoot/debug/last_build_failure.log. Set R4D_DEBUG=1 to
// mirror the same block to stderr.
// BuildExecutable parses, checks, lowers to MIR → LLVM IR, and links with clang.
// If bench is non-nil, records timings for each pipeline stage (including those inside CheckFile).
func BuildExecutable(pkgRoot, roma4dPath, outExe string, bench *BuildBench) ([]string, error) {
	var all []string
	clang := FindClang()
	if clang == "" {
		WriteBuildFailureLog(pkgRoot, "find_clang", [][2]string{
			{"error", "no clang on PATH"},
			{"hint", "install LLVM and ensure clang is on PATH; on Windows add MinGW-w64 bin too"},
		})
		return nil, fmt.Errorf("BuildExecutable: no `clang` on PATH (install LLVM for native builds)")
	}

	cr, err := CheckFile(pkgRoot, roma4dPath, bench)
	if err != nil {
		WriteBuildFailureLog(pkgRoot, "check_file", [][2]string{
			{"roma4dPath", roma4dPath},
			{"error", err.Error()},
		})
		return nil, fmt.Errorf("%s: %w", roma4dPath, err)
	}
	if len(cr.Errors) > 0 {
		msg := strings.Join(cr.Errors, "\n")
		WriteBuildFailureLog(pkgRoot, "typecheck", [][2]string{
			{"roma4dPath", roma4dPath},
			{"errors", msg},
		})
		return nil, fmt.Errorf("%s: type/ownership errors: %s", roma4dPath, strings.Join(cr.Errors, "; "))
	}

	t := time.Now()
	mir, mErrs := LowerToMIR(cr)
	if bench != nil {
		bench.Add("lower_ast_to_mir", time.Since(t))
	}
	if len(mErrs) > 0 {
		msg := strings.Join(mErrs, "\n")
		WriteBuildFailureLog(pkgRoot, "lower_mir", [][2]string{
			{"roma4dPath", roma4dPath},
			{"errors", msg},
		})
		return nil, fmt.Errorf("LowerToMIR: %s", strings.Join(mErrs, "; "))
	}

	t = time.Now()
	mod, w, err := LowerMIRToLLVM(mir)
	if bench != nil {
		bench.Add("lower_mir_to_llvm", time.Since(t))
	}
	all = append(all, w...)
	if err != nil {
		WriteBuildFailureLog(pkgRoot, "llvm_lower", [][2]string{
			{"roma4dPath", roma4dPath},
			{"error", err.Error()},
		})
		return all, err
	}
	t = time.Now()
	llStr, err := LLVMModuleString(mod)
	if bench != nil {
		bench.Add("llvm_module_string", time.Since(t))
	}
	if err != nil {
		WriteBuildFailureLog(pkgRoot, "llvm_string", [][2]string{
			{"error", err.Error()},
		})
		return all, err
	}

	tmpDir, err := os.MkdirTemp("", "roma4d-llvm-*")
	if err != nil {
		WriteBuildFailureLog(pkgRoot, "tmpdir", [][2]string{{"error", err.Error()}})
		return all, err
	}
	defer os.RemoveAll(tmpDir)

	llPath := filepath.Join(tmpDir, "out.ll")
	objPath := filepath.Join(tmpDir, "out.o")
	t = time.Now()
	if err := os.WriteFile(llPath, []byte(llStr), 0o644); err != nil {
		WriteBuildFailureLog(pkgRoot, "write_ll", [][2]string{{"error", err.Error()}, {"llPath", llPath}})
		return all, err
	}
	if bench != nil {
		bench.Add("write_ll_file", time.Since(t))
	}

	compileArgs := append([]string(nil), windowsGNUChain()...)
	compileArgs = append(compileArgs, windowsClangTarget()...)
	compileArgs = append(compileArgs, "-c", "-O1", "-o", objPath, llPath)
	compile := exec.Command(clang, compileArgs...)
	compile.Env = os.Environ()
	t = time.Now()
	if out, e := compile.CombinedOutput(); e != nil {
		WriteBuildFailureLog(pkgRoot, "clang_compile", [][2]string{
			{"clang", clang},
			{"command", formatArgv0(clang, compileArgs)},
			{"tmpDir", tmpDir},
			{"llPath", llPath},
			{"ll_head", readFileHead(llPath, 12000)},
			{"clang_output", string(out)},
			{"error", e.Error()},
		})
		return all, fmt.Errorf("clang -c failed: %w\n%s", e, string(out))
	}
	if bench != nil {
		bench.Add("clang_compile_ll", time.Since(t))
	}

	linkArgs := append([]string(nil), windowsGNUChain()...)
	linkArgs = append(linkArgs, windowsClangTarget()...)
	linkArgs = append(linkArgs, "-o", outExe, objPath)
	rt := filepath.Join(pkgRoot, "rt", "roma4d_rt.c")
	rtFiles := []string{}
	if st, e := os.Stat(rt); e == nil && !st.IsDir() {
		linkArgs = append(linkArgs, rt)
		rtFiles = append(rtFiles, rt)
	}
	if mir.HasGPUParSpacetime() {
		stub := filepath.Join(pkgRoot, "rt", "roma4d_cuda_stub.c")
		if st, e := os.Stat(stub); e == nil && !st.IsDir() {
			linkArgs = append(linkArgs, stub)
			rtFiles = append(rtFiles, stub)
		}
	}
	if runtime.GOOS != "windows" {
		linkArgs = append(linkArgs, "-lm")
	}
	link := exec.Command(clang, linkArgs...)
	link.Env = os.Environ()
	t = time.Now()
	if out, e := link.CombinedOutput(); e != nil {
		WriteBuildFailureLog(pkgRoot, "clang_link", [][2]string{
			{"clang", clang},
			{"command", formatArgv0(clang, linkArgs)},
			{"outExe", outExe},
			{"objPath", objPath},
			{"rt_sources", strings.Join(rtFiles, "\n")},
			{"tmpDir", tmpDir},
			{"ll_head", readFileHead(llPath, 12000)},
			{"clang_output", string(out)},
			{"error", e.Error()},
		})
		err := fmt.Errorf("clang link failed: %w\n%s", e, string(out))
		if runtime.GOOS == "windows" && windowsClangTarget() != nil {
			err = fmt.Errorf("%w\nhint (Windows): Roma4D uses -target *-pc-windows-gnu so Clang links with MinGW, not MSVC. Install MinGW-w64 (e.g. MSYS2: mingw-w64-ucrt-x86_64-gcc) and put its bin on PATH, or install Visual Studio Build Tools if you prefer the MSVC toolchain\nfull diagnostics: %s", err, filepath.Join(pkgRoot, "debug", "last_build_failure.log"))
		} else {
			err = fmt.Errorf("%w\nfull diagnostics: %s", err, filepath.Join(pkgRoot, "debug", "last_build_failure.log"))
		}
		return all, err
	}
	if bench != nil {
		bench.Add("clang_link_exe", time.Since(t))
	}

	return all, nil
}
