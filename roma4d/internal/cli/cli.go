package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

// Version matches roma4d.toml; bump together when releasing.
const Version = "0.1.0"

// Main is the shared entry for both `r4d` and `roma4d` binaries.
func Main(argv []string) int {
	if len(argv) < 2 {
		usage()
		return 1
	}
	switch argv[1] {
	case "-h", "--help", "help":
		usage()
		return 0
	case "version", "--version", "-V":
		fmt.Printf("roma4d (r4d) %s %s/%s\n", Version, runtime.GOOS, runtime.GOARCH)
		return 0
	case "build":
		return cmdBuild(argv[0], argv[2:])
	case "run":
		return cmdRun(argv[0], argv[2:])
	default:
		fmt.Fprintf(os.Stderr, "r4d: unknown command %q\n\n", argv[1])
		usage()
		return 1
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `roma4d — the first 4D spacetime programming language

Usage:
  r4d [command]
  roma4d [command]

Commands:
  build <file.roma4d> [-o path] [-bench]   Compile (needs clang on PATH); -bench prints pipeline timings
  run   <file.roma4d> [-bench] [args...]   Build temp binary, run; -bench adds native_run ms
  version                         Print toolchain version
  help, --help                    Show this message

Examples:
  r4d run examples/hello_4d.roma4d
  r4d run -bench examples/hello_4d.roma4d
  r4d build examples/hello_4d.roma4d -o ./hello_demo

PowerShell: type only the command line, not the leading "PS C:\...>" prompt
(PS is an alias for Get-Process and will break pasted lines).

Use a path to a real .roma4d file inside a folder tree that contains roma4d.toml
(e.g. cd to your Roma4D clone first, then: r4d run examples\hello_4d.roma4d).
`)
}

// toolLabel returns "r4d" or "roma4d" from argv0 for user-facing messages.
func toolLabel(argv0 string) string {
	b := filepath.Base(argv0)
	b = strings.TrimSuffix(b, filepath.Ext(b))
	if b == "" {
		return "r4d"
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

// ensureSourceFile checks that abs is an existing regular file before we search for roma4d.toml.
func ensureSourceFile(abs string) error {
	fi, err := os.Stat(abs)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("source file not found: %s\nhint: cd to your Roma4D repo (where roma4d.toml lives) or pass a full path to the .roma4d file", abs)
		}
		return err
	}
	if fi.IsDir() {
		return fmt.Errorf("expected a .roma4d file, not a directory: %s", abs)
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
			return "", fmt.Errorf("roma4d.toml not found above %s\nhint: run from inside the Roma4D project, or pass the full path to examples\\hello_4d.roma4d under that tree", from)
		}
		dir = parent
	}
}

func cmdBuild(argv0 string, args []string) int {
	tool := toolLabel(argv0)
	outPath := ""
	bench := false
	var pos []string
	for i := 0; i < len(args); i++ {
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
		fmt.Fprintf(os.Stderr, "%s build: %v\n", tool, err)
		return 1
	}
	pkgRoot, err := findPkgRoot(srcAbs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s build: %v\n", tool, err)
		return 1
	}
	if outPath == "" {
		base := strings.TrimSuffix(filepath.Base(srcAbs), filepath.Ext(srcAbs))
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
		fmt.Fprintf(os.Stderr, "%s build: %v\n", tool, err)
		return 1
	}
	if b != nil {
		b.WriteReport(os.Stderr, srcAbs)
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", outPath)
	printPassedSummary(tool, "build", len(warns))
	return 0
}

func cmdRun(argv0 string, args []string) int {
	tool := toolLabel(argv0)
	bench := false
	var filtered []string
	for _, a := range args {
		if a == "-bench" {
			bench = true
			continue
		}
		filtered = append(filtered, a)
	}
	args = filtered
	if len(args) < 1 {
		fmt.Fprintf(os.Stderr, "%s run: need a .roma4d source file\n", tool)
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
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
	}
	pkgRoot, err := findPkgRoot(srcAbs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
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
		fmt.Fprintf(os.Stderr, "%s run: %v\n", tool, err)
		return 1
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
