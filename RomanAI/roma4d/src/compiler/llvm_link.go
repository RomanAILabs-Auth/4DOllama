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

// detectMingwRoot returns a MinGW prefix (directory with lib/gcc/) using R4D_GNU_ROOT,
// gcc on PATH, or common MSYS2 locations.
func detectMingwRoot() string {
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
		return filepath.Clean(root)
	}
	return ""
}

func mingwGccTriplet() string {
	switch runtime.GOARCH {
	case "arm64":
		return "aarch64-w64-mingw32"
	case "386":
		return "i686-w64-mingw32"
	default:
		return "x86_64-w64-mingw32"
	}
}

// gccVersionLess returns true if a is older than b (numeric dot segments; non-digits ignored per segment).
func gccVersionLess(a, b string) bool {
	parse := func(s string) []int {
		parts := strings.Split(s, ".")
		out := make([]int, 0, len(parts))
		for _, p := range parts {
			n := 0
			for _, r := range p {
				if r < '0' || r > '9' {
					break
				}
				n = n*10 + int(r-'0')
			}
			out = append(out, n)
		}
		return out
	}
	aa, bb := parse(a), parse(b)
	n := len(aa)
	if len(bb) > n {
		n = len(bb)
	}
	for i := 0; i < n; i++ {
		var x, y int
		if i < len(aa) {
			x = aa[i]
		}
		if i < len(bb) {
			y = bb[i]
		}
		if x != y {
			return x < y
		}
	}
	return false
}

// mingwGCCBuiltinIncludeDir returns .../lib/gcc/<triplet>/<ver>/include where mm_malloc.h lives.
// Some Clang builds ignore --gcc-toolchain when compiling C during a single link invocation;
// adding -isystem here fixes MSYS2 ucrt64 + malloc.h → mm_malloc.h failures.
func mingwGCCBuiltinIncludeDir(mingwRoot string) string {
	if mingwRoot == "" {
		return ""
	}
	pat := filepath.Join(mingwRoot, "lib", "gcc", mingwGccTriplet(), "*", "include", "mm_malloc.h")
	matches, err := filepath.Glob(pat)
	if err != nil || len(matches) == 0 {
		return ""
	}
	bestInc := ""
	bestVer := ""
	for _, m := range matches {
		// m = <root>/lib/gcc/<triplet>/<ver>/include/mm_malloc.h
		incDir := filepath.Dir(m)
		ver := filepath.Base(filepath.Dir(incDir))
		if bestInc == "" || gccVersionLess(bestVer, ver) {
			bestInc, bestVer = incDir, ver
		}
	}
	return bestInc
}

// mingwSysIncludeDir returns <prefix>/include when present (e.g. MSYS2 ucrt64/include).
func mingwSysIncludeDir(mingwRoot string) string {
	if mingwRoot == "" {
		return ""
	}
	inc := filepath.Join(mingwRoot, "include")
	if st, err := os.Stat(inc); err != nil || !st.IsDir() {
		return ""
	}
	return filepath.Clean(inc)
}

func forwardSlashWin(p string) string {
	return strings.ReplaceAll(filepath.Clean(p), `\`, `/`)
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
	root := detectMingwRoot()
	if root == "" {
		return nil
	}
	rootF := forwardSlashWin(root)
	out := []string{"--gcc-toolchain=" + rootF}
	// GCC builtin dir: mm_malloc.h (included from malloc.h).
	if inc := mingwGCCBuiltinIncludeDir(root); inc != "" {
		out = append(out, "-isystem", forwardSlashWin(inc))
	}
	// MinGW system headers (stdlib, Windows CRT glue).
	if sys := mingwSysIncludeDir(root); sys != "" {
		out = append(out, "-isystem", forwardSlashWin(sys))
	}
	return out
}

// BuildExecutable parses and checks roma4dPath, lowers to MIR → LLVM IR, then links a native exe.
//
// Windows: default linker is Zig (`zig cc` compiles .ll and links rt/roma4d_rt.c; bundled libc, no MSYS2).
//          LLVM IR from codegen_llvm.go is unchanged. If Zig is missing, falls back to clang + MinGW — see windowsGNUChain.
// Linux/macOS: LLVM `clang` only (`-lm` on link when not Windows).
//
// If rt/roma4d_rt.c exists it is linked for runtime symbols. Set R4D_DEBUG=1 to mirror failures to stderr.
func BuildExecutable(pkgRoot, roma4dPath, outExe string, bench *BuildBench) ([]string, error) {
	var all []string
	zig := ""
	useZig := false
	if runtime.GOOS == "windows" {
		zig = FindZig()
		useZig = zig != ""
	}
	clang := ""
	if !useZig {
		clang = FindClang()
		if clang == "" {
			if runtime.GOOS == "windows" {
				WriteBuildFailureLog(pkgRoot, "find_toolchain", [][2]string{
					{"error", "Zig not found on PATH (and no clang for fallback)"},
					{"hint", "Install Zig (default on Windows): winget install Zig.Zig — or https://ziglang.org/download/ — then add zig.exe to PATH. Optional: set R4D_ZIG to the full path. Fallback: LLVM clang + MinGW-w64 (MSYS2) + R4D_GNU_ROOT."},
				})
				return nil, fmt.Errorf("BuildExecutable: on Windows install Zig on PATH (default; try: winget install Zig.Zig), or set R4D_ZIG, or install LLVM clang + MinGW for fallback")
			}
			WriteBuildFailureLog(pkgRoot, "find_clang", [][2]string{
				{"error", "no clang on PATH"},
				{"hint", "install LLVM and ensure clang is on PATH"},
			})
			return nil, fmt.Errorf("BuildExecutable: no `clang` on PATH (install LLVM for native builds)")
		}
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

	rt := filepath.Join(pkgRoot, "rt", "roma4d_rt.c")
	rtFiles := []string{}
	if st, e := os.Stat(rt); e == nil && !st.IsDir() {
		rtFiles = append(rtFiles, rt)
	}
	romanaiLay := filepath.Join(pkgRoot, "rt", "romanai_gguf_layout.c")
	if st, e := os.Stat(romanaiLay); e == nil && !st.IsDir() {
		rtFiles = append(rtFiles, romanaiLay)
	}
	if mir.HasGPUParSpacetime() {
		stub := filepath.Join(pkgRoot, "rt", "roma4d_cuda_stub.c")
		if st, e := os.Stat(stub); e == nil && !st.IsDir() {
			rtFiles = append(rtFiles, stub)
		}
	}

	t = time.Now()
	if useZig {
		compileArgs := ZigCCCompileArgs(llPath, objPath)
		compile := exec.Command(zig, compileArgs...)
		compile.Env = os.Environ()
		if out, e := compile.CombinedOutput(); e != nil {
			WriteBuildFailureLog(pkgRoot, "zig_compile", [][2]string{
				{"zig", zig},
				{"command", formatArgv0(zig, compileArgs)},
				{"tmpDir", tmpDir},
				{"llPath", llPath},
				{"ll_head", readFileHead(llPath, 12000)},
				{"tool_output", string(out)},
				{"error", e.Error()},
			})
			return all, fmt.Errorf("zig cc compile (.ll) failed: %w\n%s", e, string(out))
		}
		if bench != nil {
			bench.Add("zig_compile_ll", time.Since(t))
		}

		linkArgs := ZigCCLinkArgs(outExe, objPath, rtFiles)
		link := exec.Command(zig, linkArgs...)
		link.Env = os.Environ()
		t = time.Now()
		if out, e := link.CombinedOutput(); e != nil {
			WriteBuildFailureLog(pkgRoot, "zig_link", [][2]string{
				{"zig", zig},
				{"command", formatArgv0(zig, linkArgs)},
				{"outExe", outExe},
				{"objPath", objPath},
				{"rt_sources", strings.Join(rtFiles, "\n")},
				{"tmpDir", tmpDir},
				{"ll_head", readFileHead(llPath, 12000)},
				{"tool_output", string(out)},
				{"error", e.Error()},
			})
			err := fmt.Errorf("zig cc link failed: %w\n%s", e, string(out))
			err = fmt.Errorf("%w\nhint: install Zig (winget install Zig.Zig) or set R4D_ZIG. Fallback: clang + MinGW.\nfull diagnostics: %s", err, filepath.Join(pkgRoot, "debug", "last_build_failure.log"))
			return all, err
		}
		if bench != nil {
			bench.Add("zig_link_exe", time.Since(t))
		}
		return all, nil
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
	linkArgs = append(linkArgs, rtFiles...)
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
			err = fmt.Errorf("%w\nhint (Windows clang fallback): install Zig on PATH first (default linker); or fix MinGW headers (R4D_GNU_ROOT, MSYS2 ucrt64).\nfull diagnostics: %s", err, filepath.Join(pkgRoot, "debug", "last_build_failure.log"))
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

