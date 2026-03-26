package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/RomanAILabs-Auth/Roma4D/src/ai"
	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

// Version matches roma4d.toml; bump together when releasing.
const Version = "0.1.0"

// EmbeddedPkgRoot is optionally set at link time, e.g.
//
//	go install -ldflags "-X github.com/RomanAILabs-Auth/Roma4D/internal/cli.EmbeddedPkgRoot=C:/path/to/roma4d" ./cmd/r4d
//
// When the source file is not under a directory tree that contains roma4d.toml, this value is used so
// `r4d C:\anywhere\hello.r4d` works from any working directory without setting R4D_PKG_ROOT.
var EmbeddedPkgRoot string

// Main is the shared entry for `r4`, `r4d`, and `roma4d` binaries.
func Main(argv []string) int {
	if len(argv) < 2 {
		usage()
		return 1
	}
	strict, rest := stripCliModeFlags(argv[1:])
	if len(rest) == 0 {
		usage()
		return 1
	}
	switch rest[0] {
	case "-h", "--help", "help":
		usage()
		return 0
	case "version", "--version", "-V":
		fmt.Printf("roma4d (r4d) %s %s/%s\n", Version, runtime.GOOS, runtime.GOARCH)
		return 0
	case "build":
		return cmdBuild(argv[0], rest[1:], strict)
	case "run":
		return cmdRun(argv[0], rest[1:], strict)
	default:
		abs, kind, err := resolveLaunchTarget(rest[0])
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", toolLabel(argv[0]), err)
			return 1
		}
		switch kind {
		case launchPython:
			return runPythonScript(argv[0], abs, rest[1:])
		case launchR4D:
			// Shorthand: r4d file.r4d  (same as run; expert hints on failure unless --strict)
			newRest := append([]string{abs}, rest[1:]...)
			return cmdRun(argv[0], newRest, strict)
		default:
			fmt.Fprintf(os.Stderr, "%s: unknown command or script %q\n\n", toolLabel(argv[0]), rest[0])
			usage()
			return 1
		}
	}
}

type launchKind int

const (
	launchNone launchKind = iota
	launchR4D
	launchPython
)

// resolveLaunchTarget maps a user path to an absolute .r4d / .py file.
// Bare names (no extension) try Roma4D extensions first (.r4d, .r4s, .roma4d), then .py / .pyw.
func resolveLaunchTarget(arg string) (abs string, kind launchKind, err error) {
	if strings.HasPrefix(arg, "-") {
		return "", launchNone, fmt.Errorf("expected a script path, got flag %q", arg)
	}
	arg = filepath.Clean(arg)
	ext := strings.ToLower(filepath.Ext(arg))

	if ext == ".py" || ext == ".pyw" {
		abs, err = filepath.Abs(arg)
		if err != nil {
			return "", launchNone, err
		}
		fi, e := os.Stat(abs)
		if e != nil {
			if os.IsNotExist(e) {
				return "", launchNone, fmt.Errorf("Python script not found: %s", abs)
			}
			return "", launchNone, e
		}
		if fi.IsDir() {
			return "", launchNone, fmt.Errorf("expected a .py file, not a directory: %s", abs)
		}
		return abs, launchPython, nil
	}

	if compiler.IsRoma4DSourcePath(arg) {
		abs, err = filepath.Abs(arg)
		if err != nil {
			return "", launchNone, err
		}
		fi, e := os.Stat(abs)
		if e != nil {
			if os.IsNotExist(e) {
				return "", launchNone, fmt.Errorf("Roma4D source not found: %s", abs)
			}
			return "", launchNone, e
		}
		if fi.IsDir() {
			return "", launchNone, fmt.Errorf("expected a Roma4D source file, not a directory: %s", abs)
		}
		return abs, launchR4D, nil
	}

	if ext != "" {
		return "", launchNone, fmt.Errorf("unknown extension %q (use .r4d, .py, or omit extension to pick .r4d then .py)", ext)
	}

	absBase, err := filepath.Abs(arg)
	if err != nil {
		return "", launchNone, err
	}
	if fi, e := os.Stat(absBase); e == nil && fi.IsDir() {
		return "", launchNone, fmt.Errorf("expected a script path, not a directory: %s", absBase)
	}
	for _, e := range compiler.Roma4DSourceExtensions {
		p := absBase + e
		fi, e2 := os.Stat(p)
		if e2 == nil && !fi.IsDir() {
			return p, launchR4D, nil
		}
	}
	for _, e := range []string{".py", ".pyw"} {
		p := absBase + e
		fi, e2 := os.Stat(p)
		if e2 == nil && !fi.IsDir() {
			return p, launchPython, nil
		}
	}
	base := filepath.Base(absBase)
	return "", launchNone, fmt.Errorf("no script found for %q (looked for %s.r4d, legacy .r4s/.roma4d, then %s.py)", arg, base, base)
}

// stripCliModeFlags removes --strict / --forgiving from args. Default is forgiving (expert hints on).
// If both appear, the later one wins.
func stripCliModeFlags(args []string) (strict bool, out []string) {
	strict = false
	for _, a := range args {
		switch a {
		case "--strict":
			strict = true
		case "--forgiving":
			strict = false
		default:
			out = append(out, a)
		}
	}
	return strict, out
}

func usage() {
	fmt.Fprintf(os.Stderr, `roma4d — the first 4D spacetime programming language

The r4d and roma4d commands behave the same: one launcher for Python and for native Roma4D.

Run a script (Python-like ergonomics):
  r4d myapp.py [args...]              Run with the system Python (PyQt, torch, numpy, …)
  r4d myapp.r4d [args...]             Compile + run native high-performance Roma4D
  r4d myapp [args...]                 If myapp has no extension: try myapp.r4d, then myapp.py

Same as Python: you can pass a path with or without extension; Roma4D sources win when both exist.

Usage:
  r4d [--strict|--forgiving] <file.py|.r4d> [args...]   Python or native (see above)
  r4d [--strict|--forgiving] run <file> [args...]       Same resolution rules for the script
  r4d [--strict|--forgiving] build <file.r4d> [-o path] [-bench]

Flags:
  --strict      For .r4d only: disable Native AI Expert (no rich debug block / interactive session).
  --forgiving   For .r4d only: enable Expert (default). Ignored when running .py.

Environment (Expert):
  R4D_EXPERT_INTERACTIVE=0   Print the debug block but skip patch prompt + interactive REPL (e.g. pipes, CI).
  R4D_DEBUG=1                Mirror linker failure details to stderr (see Guide §19).

Environment (Windows native .r4d builds — LLVM IR unchanged, linker driver only):
  R4D_ZIG                    Full path to zig.exe if not on PATH (default linker: zig cc).

Commands:
  build   Native compile (.r4d): Windows uses Zig (zig cc) by default; clang+MinGW only if Zig missing. Unix/macOS: clang.
  run     Same as build for .r4d; .py runs under Python
  version Print toolchain version
  help    Show this message

Source extensions: .r4d (official). Legacy: .r4s, .roma4d. Python: .py, .pyw.

When you run a .py file, r4d prints a short note that .r4d gives native 4D acceleration.

Examples:
  r4d tool.py
  r4d demos/hello.r4d
  r4d mytool              # picks mytool.r4d if present, else mytool.py
  r4d --strict run examples/min_main.r4d
  r4d run -bench examples/hello_4d.r4d

PowerShell: type only the command line, not the leading "PS C:\...>" prompt
(PS is an alias for Get-Process and will break pasted lines).

You can run a program from any folder: pass a path to the .r4d file (absolute is safest for one-off locations).

Package root (stdlib / roma4d.toml): we walk upward from the source file first. If that fails, we use
R4D_PKG_ROOT or ROMA4D_HOME. Binaries built by .\scripts\Install-R4dUserEnvironment.ps1 (or install.ps1 /
install.sh) also embed that repo path, so you do not have to cd into the Roma4D clone for normal use.

Windows one-shot setup (PATH + R4D_PKG_ROOT + embedded root):  .\scripts\Install-R4dUserEnvironment.ps1
  Install Zig first for the default linker (no MSYS2 required):  winget install Zig.Zig
`)
}

// toolLabel returns "r4", "r4d", or "roma4d" from argv0 for user-facing messages.
func toolLabel(argv0 string) string {
	b := filepath.Base(argv0)
	b = strings.TrimSuffix(b, filepath.Ext(b))
	if b == "" {
		return "r4d"
	}
	if b == "r4" {
		return "r4"
	}
	return b
}

func printCompilerWarnings(warns []string) {
	for _, w := range warns {
		fmt.Fprintf(os.Stderr, "warning: %s\n", w)
	}
}

// printPassedSummary prints a friendly outcome line after a successful build or run.
func printPassedSummary(tool, verb string, nWarn int) {
	if nWarn == 0 {
		fmt.Fprintf(os.Stderr, "%s %s: passed.\n", tool, verb)
		return
	}
	w := "warnings"
	if nWarn == 1 {
		w = "warning"
	}
	fmt.Fprintf(os.Stderr, "%s %s: passed (with %d %s).\n", tool, verb, nWarn, w)
}

// ensureSourceFile checks that abs is an existing Roma4D source file before we search for roma4d.toml.
func ensureSourceFile(abs string) error {
	fi, err := os.Stat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("source file not found: %s\nhint: check spelling. If the file is on your Desktop, use the full path, for example:\n  r4d C:\\Users\\You\\Desktop\\myfile.r4d", abs)
		}
		return err
	}
	if fi.IsDir() {
		return fmt.Errorf("expected a .r4d source file, not a directory: %s", abs)
	}
	if !compiler.IsRoma4DSourcePath(abs) {
		return fmt.Errorf("not a Roma4D source file (use .r4d, or legacy .r4s / .roma4d): %s", abs)
	}
	return nil
}

func findPkgRoot(from string) (string, error) {
	dir := from
	if fi, err := os.Stat(from); err == nil && !fi.IsDir() {
		dir = filepath.Dir(from)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for {
		toml := filepath.Join(dir, "roma4d.toml")
		if _, err := os.Stat(toml); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	for _, key := range []string{"R4D_PKG_ROOT", "ROMA4D_HOME"} {
		raw := strings.TrimSpace(os.Getenv(key))
		if raw == "" {
			continue
		}
		root, err := filepath.Abs(raw)
		if err != nil {
			continue
		}
		toml := filepath.Join(root, "roma4d.toml")
		if st, err := os.Stat(toml); err == nil && !st.IsDir() {
			return root, nil
		}
	}
	rawEmb := strings.TrimSpace(EmbeddedPkgRoot)
	if rawEmb != "" {
		root, err := filepath.Abs(rawEmb)
		if err == nil {
			toml := filepath.Join(root, "roma4d.toml")
			if st, err := os.Stat(toml); err == nil && !st.IsDir() {
				return root, nil
			}
		}
	}
	return "", fmt.Errorf("could not find Roma4D installation (roma4d.toml) for:\n  %s\n\nFix: run once from your Roma4D folder, then open a NEW terminal:\n  .\\scripts\\Install-R4dUserEnvironment.ps1\nAfter that, r4d yourfile.r4d works from anywhere.", from)
}

func reportBuildFailure(tool, verb, srcAbs string, err error, strict bool) int {
	raw := err.Error()
	if !strict {
		pkgRoot, _ := findPkgRoot(srcAbs)
		return ai.HandleFailure(ai.FailureContext{
			Tool: tool, Verb: verb,
			SourcePath:  srcAbs,
			PackageRoot: pkgRoot,
			RawError:    raw,
			Stdin:       os.Stdin,
			Stdout:      os.Stdout,
			Stderr:      os.Stderr,
		})
	}
	fmt.Fprintf(os.Stderr, "%s %s: %v\n", tool, verb, err)
	return 1
}

func cmdBuild(argv0 string, args []string, strict bool) int {
	tool := toolLabel(argv0)
	outPath := ""
	bench := false
	var pos []string
	for i := 0; i < len(args); i++ {
		if args[i] == "--strict" {
			strict = true
			continue
		}
		if args[i] == "--forgiving" {
			strict = false
			continue
		}
		if args[i] == "-o" && i+1 < len(args) {
			outPath = args[i+1]
			i++
			continue
		}
		if args[i] == "-bench" {
			bench = true
			continue
		}
		if strings.HasPrefix(args[i], "-") {
			fmt.Fprintf(os.Stderr, "%s build: unknown flag %q\n", tool, args[i])
			return 1
		}
		pos = append(pos, args[i])
	}
	if len(pos) != 1 {
		fmt.Fprintf(os.Stderr, "%s build: need exactly one source file (use -o for output path)\n", tool)
		return 1
	}
	src := pos[0]
	srcAbs, err := filepath.Abs(src)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s build: %v\n", tool, err)
		return 1
	}
	if err := ensureSourceFile(srcAbs); err != nil {
		return reportBuildFailure(tool, "build", srcAbs, err, strict)
	}
	pkgRoot, err := findPkgRoot(srcAbs)
	if err != nil {
		return reportBuildFailure(tool, "build", srcAbs, err, strict)
	}
	if outPath == "" {
		base := compiler.StripRoma4DSourceExt(filepath.Base(srcAbs))
		outPath = base
		if runtime.GOOS == "windows" {
			outPath += ".exe"
		}
	} else {
		outPath, err = filepath.Abs(outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s build: %v\n", tool, err)
			return 1
		}
	}
	var b *compiler.BuildBench
	if bench {
		b = &compiler.BuildBench{}
	}
	warns, err := compiler.BuildExecutable(pkgRoot, srcAbs, outPath, b)
	printCompilerWarnings(warns)
	if err != nil {
		return reportBuildFailure(tool, "build", srcAbs, err, strict)
	}
	if b != nil {
		b.WriteReport(os.Stderr, srcAbs)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", outPath)
	printPassedSummary(tool, "build", len(warns))
	return 0
}

func cmdRun(argv0 string, args []string, strict bool) int {
	tool := toolLabel(argv0)
	bench := false
	var filtered []string
	for _, a := range args {
		if a == "--strict" {
			strict = true
			continue
		}
		if a == "--forgiving" {
			strict = false
			continue
		}
		if a == "-bench" {
			bench = true
			continue
		}
		filtered = append(filtered, a)
	}
	args = filtered
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "%s run: need a script (.r4d, .py, or base name)\n", tool)
		return 1
	}
	abs, kind, err := resolveLaunchTarget(args[0])
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
	}
	if kind == launchPython {
		return runPythonScript(argv0, abs, args[1:])
	}
	srcAbs := abs
	progArgs := args[1:]
	if err := ensureSourceFile(srcAbs); err != nil {
		return reportBuildFailure(tool, "run", srcAbs, err, strict)
	}
	pkgRoot, err := findPkgRoot(srcAbs)
	if err != nil {
		return reportBuildFailure(tool, "run", srcAbs, err, strict)
	}
	tmp, err := os.MkdirTemp("", "r4d-run-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
	}
	defer os.RemoveAll(tmp)
	exe := filepath.Join(tmp, "a.out")
	if runtime.GOOS == "windows" {
		exe = filepath.Join(tmp, "a.exe")
	}
	var b *compiler.BuildBench
	if bench {
		b = &compiler.BuildBench{}
	}
	warns, err := compiler.BuildExecutable(pkgRoot, srcAbs, exe, b)
	printCompilerWarnings(warns)
	if err != nil {
		return reportBuildFailure(tool, "run", srcAbs, err, strict)
	}
	c := exec.Command(exe, progArgs...)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	t0 := time.Now()
	errRun := c.Run()
	if b != nil {
		if errRun == nil {
			b.Add("native_run", time.Since(t0))
			b.WriteReport(os.Stderr, srcAbs)
		} else if x, ok := errRun.(*exec.ExitError); ok {
			// Non-zero exit is still a normal finished run (e.g. min_main returns 42).
			b.Add("native_run", time.Since(t0))
			b.WriteReport(os.Stderr, srcAbs)
			printPassedSummary(tool, "run", len(warns))
			return x.ExitCode()
		} else {
			b.WriteReport(os.Stderr, srcAbs)
			fmt.Fprintf(os.Stderr, "r4d bench: native_run omitted (could not run executable)\n")
			fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, errRun)
			return 1
		}
	} else if errRun != nil {
		if x, ok := errRun.(*exec.ExitError); ok {
			return x.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, errRun)
		return 1
	}
	printPassedSummary(tool, "run", len(warns))
	return 0
}

// findPythonInterpreter returns an executable and optional argv prefix (e.g. py + [-3]).
// Override with R4D_PYTHON or PYTHON (full path to interpreter).
func findPythonInterpreter() (exe string, prefix []string, err error) {
	for _, key := range []string{"R4D_PYTHON", "PYTHON"} {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			return v, nil, nil
		}
	}
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
	return "", nil, fmt.Errorf("could not find Python (install Python 3 and/or set PYTHON or R4D_PYTHON)")
}

// runPythonScript executes scriptAbs with the system interpreter; progArgs are passed after the script path.
func runPythonScript(argv0, scriptAbs string, progArgs []string) int {
	tool := toolLabel(argv0)
	exe, prefix, err := findPythonInterpreter()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", tool, err)
		return 1
	}
	args := append(append(append([]string{}, prefix...), scriptAbs), progArgs...)
	fmt.Fprintf(os.Stderr, "%s: running as Python script (use .r4d extension for native 4D acceleration)\n", tool)
	c := exec.Command(exe, args...)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	if err := c.Run(); err != nil {
		if x, ok := err.(*exec.ExitError); ok {
			return x.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "%s: %v\n", tool, err)
		return 1
	}
	return 0
}
