// Package ai implements a native, zero-LLM "Expert" layer that enriches compiler
// diagnostics with Roma4D-specific hints (forgiving mode only).
package ai

import (
	"bufio"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Pre-compiled patterns (package init — no compile in hot path).
var (
	reExtLine = regexp.MustCompile(`\.(?:r4d|r4s|roma4d):(\d+):`)
	reLineCol = regexp.MustCompile(`:(\d+):(\d+):`)
)

// ExtractErrorLine returns the first plausible source line number from a compiler/stderr blob, or 0.
func ExtractErrorLine(raw string) int {
	if m := reExtLine.FindStringSubmatch(raw); len(m) > 1 {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
			return n
		}
	}
	if m := reLineCol.FindStringSubmatch(raw); len(m) > 1 {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
			return n
		}
	}
	return 0
}

// RunExpertMode augments rawError with Roma4D-native hints when isForgiving is true.
// On any panic, it fail-opens to the original rawError. fixed is true when hints were added.
// Hot path uses strings.Contains only; file reads use bufio.Scanner (bounded, no full-file buffer).
func RunExpertMode(filePath string, rawError string, errorLineNum int, isForgiving bool) (fixed bool, newError string) {
	defer func() {
		if recover() != nil {
			fixed = false
			newError = rawError
		}
	}()

	if !isForgiving {
		return false, rawError
	}

	line := errorLineNum
	if line <= 0 {
		line = ExtractErrorLine(rawError)
	}

	var hint strings.Builder
	// Avoid repeated Grow in tight loops; one upfront cap is enough for typical hints.
	hint.Grow(512)

	switch {
	case strings.Contains(rawError, "import * is not supported"):
		hint.WriteString("Roma4D does not support `import *`. Use explicit names: `from libgeo import bump, identity_v4`.\n")
	case strings.Contains(rawError, "ImportError"):
		hint.WriteString("Check that the module exists as `libname.r4d` (or legacy `.r4s` / `.roma4d`) under the package root next to `roma4d.toml`.\n")
	case strings.Contains(rawError, "roma4d.toml not found"):
		hint.WriteString("Set environment variable R4D_PKG_ROOT (or ROMA4D_HOME) to the folder containing `roma4d.toml`, or move the source under that tree.\n")
	case strings.Contains(rawError, "not a Roma4D source file"):
		hint.WriteString("Use extension `.r4d` (official), or legacy `.r4s` / `.roma4d`.\n")
	case strings.Contains(rawError, "source file not found"):
		hint.WriteString("Confirm the path and current working directory; prefer an absolute path.\n")
	case strings.Contains(rawError, "UseAfterMoveError") || strings.Contains(rawError, "soa field"):
		hint.WriteString("SoA fields are linear: read `cell.pos` into a local, update, then assign back `cell.pos = ...` before reading again.\n")
	case strings.Contains(rawError, "BorrowError") || strings.Contains(rawError, "mutborrow"):
		hint.WriteString("Shrink borrow regions; immutable and mutable borrows of the same name cannot overlap in one block.\n")
	case strings.Contains(rawError, "TaintError"):
		hint.WriteString("Do not assign values that flowed through `print(...)` into `soa` / linear slots.\n")
	case strings.Contains(rawError, "mm_malloc.h"):
		hint.WriteString("Windows: install Zig on PATH (Roma4D default), or use MinGW with clang: set R4D_GNU_ROOT to ucrt64/mingw64.\n")
	case strings.Contains(rawError, "zig cc") && strings.Contains(rawError, "failed"):
		hint.WriteString("Install Zig from https://ziglang.org/download/ and ensure `zig` is on PATH, or set R4D_ZIG to the zig executable.\n")
	case strings.Contains(rawError, "no zig and no clang"):
		hint.WriteString("Windows: add Zig to PATH (recommended) or install LLVM clang plus MinGW-w64.\n")
	case strings.Contains(rawError, "clang") && strings.Contains(rawError, "failed"):
		hint.WriteString("See `debug/last_build_failure.log` under the package root; set R4D_DEBUG=1 to mirror diagnostics to stderr.\n")
	case strings.Contains(rawError, "synthesized return"):
		hint.WriteString("For `def main() -> int:` add an explicit `return` on all paths. For `-> None`, you may use `return None` or omit.\n")
	case strings.Contains(rawError, "unexpected token") || strings.Contains(rawError, "parse"):
		hint.WriteString("This is not Python: no f-strings, no `import os`, no `time()` call — use `t`, `vec4`, `from libgeo import ...`. See docs/Roma4D_Guide.md §21.\n")
	}

	if hint.Len() == 0 && (strings.Contains(rawError, "type") || strings.Contains(rawError, "ownership")) {
		hint.WriteString("Compare your code with `examples/hello_4d.r4d` and the LLM checklist in docs/Roma4D_Guide.md §18.\n")
	}

	if hint.Len() == 0 {
		return false, rawError
	}

	snippet := readLineWindow(filePath, line)
	var b strings.Builder
	b.Grow(len(rawError) + hint.Len() + len(snippet) + 64)
	b.WriteString(rawError)
	b.WriteString("\n\n--- Roma4D Expert (native, forgiving mode) ---\n")
	b.WriteString(hint.String())
	if snippet != "" {
		b.WriteString("Context near line ")
		b.WriteString(strconv.Itoa(line))
		b.WriteString(":\n")
		b.WriteString(snippet)
	}
	b.WriteString("---\n")
	b.WriteString("Tip: use `r4d --strict ...` or `r4d run --strict ...` to disable these hints.\n")
	return true, b.String()
}

// readLineWindow returns a small text window around 1-based line (max ~3 lines) using a Scanner only.
func readLineWindow(path string, centerLine int) string {
	if centerLine <= 0 || path == "" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	const maxScan = 512 * 1024
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, maxScan)

	start := centerLine - 1
	if start < 1 {
		start = 1
	}
	end := centerLine + 1

	var b strings.Builder
	n := 0
	for sc.Scan() {
		n++
		if n < start {
			continue
		}
		if n > end {
			break
		}
		b.WriteString(strconv.Itoa(n))
		b.WriteString(" | ")
		b.Write(sc.Bytes())
		b.WriteByte('\n')
	}
	if b.Len() == 0 {
		return ""
	}
	return b.String()
}
