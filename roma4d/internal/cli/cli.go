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
	strict, rest := stripStrictFlags(argv[1:])
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
		if compiler.IsRoma4DSourcePath(rest[0]) {
			// Forgiving shorthand: r4d file.r4d  (same as run, expert hints on failure)
			return cmdRun(argv[0], rest, strict)
		}
		fmt.Fprintf(os.Stderr, "r4d: unknown command %q\n\n", rest[0])
		usage()
		return 1
	}
}

// stripStrictFlags removes every --strict from args (any position).
func stripStrictFlags(args []string) (strict bool, out []string) {
	for _, a := range args {
		if a == "--strict" {
			strict = true
			continue
		}
		out = append(out, a)
	}
	return strict, out
}

func usage() {
	fmt.Fprintf(os.Stderr, `roma4d — the first 4D spacetime programming language

Simplest way to run your program (from any folder, after one-time Windows setup):
  r4d myfile.r4d
  r4d C:\path\to\myfile.r4d

That is all you need. Same as: r4d run <file.r4d>

Usage:
  r4d [--strict] <file.r4d> [args...]     Shorthand for run (forgiving Expert mode by default)
  r4d [--strict] run <file.r4d> [args...]
  r4d [--strict] build <file.r4d> [-o path] [-bench]

Flags:
  --strict    Disable native Expert hints; print raw compiler errors only (CI / scripting).

Commands:
  build   Native compile: Windows uses Zig (zig cc) when on PATH, else LLVM clang + MinGW; Unix uses clang. -bench prints timings
  run     Build temp binary, run; -bench adds native_run ms
  version Print toolchain version
  help    Show this message

Source extension: .r4d (official). Legacy: .r4s, .roma4d.

Forgiving mode (default): on compile failure, a native rules engine may append Roma4D-specific hints
(no external LLM, zero network). Use --strict to disable.

Examples:
  r4d demos/hello.r4d
  r4d --strict run examples/min_main.r4d
  r4d run -bench examples/hello_4d.r4d

PowerShell: type only the command line, not the leading "PS C:\...>" prompt
(PS is an alias for Get-Process and will break pasted lines).

You can run a program from any folder: pass a path to the .r4d file (absolute is safest for one-off locations).

Package root (stdlib / roma4d.toml): we walk upward from the source file first. If that fails, we use
R4D_PKG_ROOT or ROMA4D_HOME. Binaries built by .\scripts\Install-R4dUserEnvironment.ps1 (or install.ps1 /
install.sh) also embed that repo path, so you do not have to cd into the Roma4D clone for normal use.

Windows one-shot setup (PATH + R4D_PKG_ROOT + embedded root):  .\scripts\Install-R4dUserEnvironment.ps1
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
		line := ai.ExtractErrorLine(raw)
		fixed, msg := ai.RunExpertMode(srcAbs, raw, line, true)
		if fixed {
			fmt.Fprintf(os.Stderr, "%s %s:\n%s", tool, verb, msg)
			return 1
		}
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
		if a == "-bench" {
			bench = true
			continue
		}
		filtered = append(filtered, a)
	}
	args = filtered
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "%s run: need a .r4d source file (or legacy .r4s / .roma4d)\n", tool)
		return 1
	}
	src := args[0]
	progArgs := args[1:]
	srcAbs, err := filepath.Abs(src)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
	}
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
