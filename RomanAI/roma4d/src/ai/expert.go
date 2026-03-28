// Package ai implements a native, zero-network "Expert" layer: rich terminal debug output,
// copy-paste commands, LLM-oriented briefings, and optional interactive Q&A (forgiving mode).
package ai

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
	"golang.org/x/term"
)

// --- Pre-compiled patterns (package level; no regexp.Compile in hot path) ---

var (
	reExtLine     = regexp.MustCompile(`\.(?:r4d|r4s|roma4d):(\d+):`)
	reLineCol     = regexp.MustCompile(`:(\d+):(\d+):`)
	rePathDrive   = regexp.MustCompile(`(?i)^[a-z]:\\`)
	reWhy         = regexp.MustCompile(`(?i)\b(why|reason|fail|failed|error)\b`)
	reZig         = regexp.MustCompile(`(?i)zig|lld|link\.exe|linker`)
	reClang       = regexp.MustCompile(`(?i)clang|mm_malloc|mingw|msys`)
	rePython      = regexp.MustCompile(`(?i)python|pyqt|pip|numpy|torch`)
	reImport      = regexp.MustCompile(`(?i)import|module|libgeo|roma4d\.toml`)
	reOwnership   = regexp.MustCompile(`(?i)UseAfterMove|BorrowError|TaintError|soa|ownership|linear`)
	reParse       = regexp.MustCompile(`(?i)parse|INDENT|DEDENT|unexpected token`)
	reGuide       = regexp.MustCompile(`(?i)guide|manual|doc|reference`)
)

const (
	debugBannerTop    = "================ ROMA4D EXPERT DEBUG (copy from line below to end of block) ================"
	debugBannerBottom = "================ END ROMA4D EXPERT DEBUG ======================================================="
)

// FailureContext carries everything needed to render a session (CLI or tests).
type FailureContext struct {
	Tool, Verb  string
	SourcePath  string
	PackageRoot string
	RawError    string
	ErrorLine   int // 0 = auto from RawError

	Stdin  *os.File
	Stdout *os.File
	Stderr *os.File
}

func (c *FailureContext) stderr() *os.File {
	if c != nil && c.Stderr != nil {
		return c.Stderr
	}
	return os.Stderr
}

func (c *FailureContext) stdin() *os.File {
	if c != nil && c.Stdin != nil {
		return c.Stdin
	}
	return os.Stdin
}

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

// HandleFailure prints a full expert report to stderr, appends to debug/last_build_failure.log,
// optionally prompts for patch acknowledgment, then runs an interactive expert REPL when stdin is a TTY.
// Always returns 1 (failure exit code for the CLI).
func HandleFailure(ctx FailureContext) (exitCode int) {
	exitCode = 1
	defer func() {
		if r := recover(); r != nil {
			_, _ = fmt.Fprintf(ctx.stderr(), "(Roma4D Expert: recovered internal error — showing raw failure only)\n%v\n%s %s: %s\n",
				r, ctx.Tool, ctx.Verb, ctx.RawError)
		}
	}()

	if ctx.Tool == "" {
		ctx.Tool = "r4d"
	}
	if ctx.Verb == "" {
		ctx.Verb = "build"
	}
	line := ctx.ErrorLine
	if line <= 0 {
		line = ExtractErrorLine(ctx.RawError)
	}
	ctx.ErrorLine = line

	report := buildExpertReport(&ctx)
	_, _ = fmt.Fprint(ctx.stderr(), report)

	compiler.WriteBuildFailureLog(ctx.PackageRoot, "ai_expert", [][2]string{
		{"expert_terminal_report", report},
		{"raw_error", ctx.RawError},
		{"source_path", ctx.SourcePath},
		{"package_root", ctx.PackageRoot},
	})

	if expertInteractiveEnabled(&ctx) {
		patch := suggestedPatchText(&ctx)
		if patch != "" && ctx.SourcePath != "" {
			_, _ = fmt.Fprintf(ctx.stderr(), "\n--- Suggested edit (paste into %s) ---\n%s\n--- end suggested edit ---\n",
				ctx.SourcePath, patch)
			promptPatchAck(&ctx)
		}
		runInteractiveExpert(&ctx)
	}
	return exitCode
}

func expertInteractiveEnabled(ctx *FailureContext) bool {
	if strings.TrimSpace(os.Getenv("R4D_EXPERT_INTERACTIVE")) == "0" {
		return false
	}
	fd := int(ctx.stdin().Fd())
	return term.IsTerminal(fd)
}

// RunExpertMode augments rawError with a full native expert report when isForgiving is true.
// On panic, fail-opens to the original rawError. fixed is true when a report was produced.
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
	ctx := FailureContext{
		Tool: "r4d", Verb: "compile",
		SourcePath: filePath, RawError: rawError, ErrorLine: line,
		Stderr: os.Stderr,
	}
	report := buildExpertReport(&ctx)
	if !strings.Contains(report, debugBannerTop) {
		return false, rawError
	}
	return true, report
}

// buildExpertReport assembles the rich block (no interactive parts).
func buildExpertReport(ctx *FailureContext) string {
	raw := ctx.RawError
	line := ctx.ErrorLine
	if line <= 0 {
		line = ExtractErrorLine(raw)
	}

	var b strings.Builder
	b.Grow(4096)

	b.WriteString(debugBannerTop)
	b.WriteByte('\n')
	b.WriteString(fmt.Sprintf("timestamp_utc: %s\n", time.Now().UTC().Format(time.RFC3339)))
	b.WriteString(fmt.Sprintf("host: %s/%s  go: %s\n", runtime.GOOS, runtime.GOARCH, runtime.Version()))
	b.WriteString(fmt.Sprintf("tool: %s  action: %s\n", ctx.Tool, ctx.Verb))
	if ctx.SourcePath != "" {
		b.WriteString(fmt.Sprintf("source_file: %s\n", ctx.SourcePath))
	}
	if ctx.PackageRoot != "" {
		b.WriteString(fmt.Sprintf("package_root: %s\n", ctx.PackageRoot))
	}
	b.WriteString(fmt.Sprintf("inferred_line: %d\n", line))
	b.WriteString("\n--- raw compiler / driver message ---\n")
	b.WriteString(raw)
	b.WriteString("\n--- end raw message ---\n\n")

	snippet := readLineWindow(ctx.SourcePath, line)
	if snippet != "" {
		b.WriteString("--- source context (nearest lines) ---\n")
		b.WriteString(snippet)
		b.WriteString("--- end context ---\n\n")
	}

	b.WriteString(symptomHints(raw))
	b.WriteString(commandRecipes(ctx, raw))
	b.WriteString(guideMemory())
	b.WriteString(llmBriefing(ctx, raw, line))
	b.WriteString("\n")
	b.WriteString(debugBannerBottom)
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("Tip: `%s --strict ...` disables Expert output. Set R4D_EXPERT_INTERACTIVE=0 to skip the post-mortem prompt. R4D_DEBUG=1 mirrors linker logs to stderr.\n",
		ctx.Tool))
	return b.String()
}

func symptomHints(raw string) string {
	var h strings.Builder
	h.WriteString("--- symptom → native hint (Roma4D Programming Guide) ---\n")
	switch {
	case strings.Contains(raw, "import * is not supported"):
		h.WriteString("• import *: Roma4D forbids star-imports. Use explicit names, e.g. `from libgeo import bump, identity_v4` (Guide §4, §27).\n")
	case reImport.MatchString(raw) && strings.Contains(raw, "ImportError"):
		h.WriteString("• Import: ensure `modulename.r4d` exists under the package root (folder with `roma4d.toml`). Imports are not PyPI (Guide §4).\n")
	case strings.Contains(raw, "roma4d.toml not found") || strings.Contains(raw, "could not find Roma4D installation"):
		h.WriteString("• Package root: set R4D_PKG_ROOT or ROMA4D_HOME to the repo root, or run Install-R4dUserEnvironment.ps1 once (Guide §3, §24).\n")
	case strings.Contains(raw, "not a Roma4D source file"):
		h.WriteString("• Extension: use `.r4d` (or legacy `.r4s` / `.roma4d`). For Python use `r4d script.py` (Guide §21).\n")
	case strings.Contains(raw, "source file not found"):
		h.WriteString("• Path: confirm spelling and cwd; prefer an absolute path to the `.r4d` file.\n")
	case strings.Contains(raw, "UseAfterMoveError") || strings.Contains(raw, "soa field"):
		h.WriteString("• SoA linear field: read `cell.pos` into a local, update, assign back before reading again (Guide §11).\n")
	case strings.Contains(raw, "BorrowError") || strings.Contains(raw, "mutborrow"):
		h.WriteString("• Borrows: shrink regions; immutable and mutable borrows of the same name must not overlap (Guide §11).\n")
	case strings.Contains(raw, "TaintError"):
		h.WriteString("• Taint: do not assign values that flowed through `print(...)` into `soa` / linear slots (Guide §11).\n")
	case strings.Contains(raw, "mm_malloc.h"):
		h.WriteString("• Windows + clang: MinGW internal headers missing — install Zig on PATH (recommended) or set R4D_GNU_ROOT to MSYS2 ucrt64 (Guide §19).\n")
	case strings.Contains(raw, "zig cc") && strings.Contains(raw, "failed"):
		h.WriteString("• Zig: install from https://ziglang.org/download/ and add to PATH, or set R4D_ZIG (Guide §1, §19).\n")
	case strings.Contains(raw, "no zig and no clang"):
		h.WriteString("• Toolchain: Windows needs Zig (simplest) or LLVM clang + MinGW-w64 (Guide §1).\n")
	case strings.Contains(raw, "clang") && strings.Contains(raw, "failed"):
		h.WriteString("• Link/compile: see `debug/last_build_failure.log` under package root; enable R4D_DEBUG=1 (Guide §19).\n")
	case strings.Contains(raw, "synthesized return"):
		h.WriteString("• main(): add explicit `return` on all paths for `-> int`; `-> None` may omit or `return None` (Guide §5, §27).\n")
	case reParse.MatchString(raw):
		h.WriteString("• Parse: Roma4D is not Python — no f-strings, no `import os`, no `time()`; see Guide §22, §27.\n")
	default:
		if strings.Contains(raw, "type") || strings.Contains(raw, "ownership") || reOwnership.MatchString(raw) {
			h.WriteString("• Types/ownership: compare with `examples/hello_4d.r4d` and checklist §26.\n")
		} else {
			h.WriteString("• No exact rule matched; use the command recipes below and the LLM briefing block.\n")
		}
	}
	h.WriteString("--- end hints ---\n\n")
	return h.String()
}

func commandRecipes(ctx *FailureContext, raw string) string {
	var b strings.Builder
	b.WriteString("--- copy/paste commands (adjust paths) ---\n")
	src := ctx.SourcePath
	if src == "" {
		src = filepath.Join("path", "to", "file.r4d")
	}
	absHint := src
	if !rePathDrive.MatchString(absHint) && runtime.GOOS == "windows" {
		absHint = `C:\full\path\to\file.r4d`
	}

	b.WriteString(fmt.Sprintf("  # Re-run with full path:\n  %s %s\n", ctx.Tool, absHint))
	b.WriteString(fmt.Sprintf("  # Strict (CI-style, no Expert):\n  %s --strict %s\n", ctx.Tool, absHint))
	b.WriteString("  # Verbose linker mirror:\n")
	if runtime.GOOS == "windows" {
		b.WriteString(fmt.Sprintf("  $env:R4D_DEBUG=\"1\"; %s %s\n", ctx.Tool, absHint))
	} else {
		b.WriteString(fmt.Sprintf("  R4D_DEBUG=1 %s %s\n", ctx.Tool, absHint))
	}

	if reZig.MatchString(raw) || strings.Contains(raw, "no zig") {
		if runtime.GOOS == "windows" {
			b.WriteString("  # Install Zig (Windows):\n")
			b.WriteString("  winget install --id Zig.Zig -e\n")
			b.WriteString("  # Or: scoop install zig\n")
		} else if runtime.GOOS == "darwin" {
			b.WriteString("  # macOS Zig:\n  brew install zig\n")
		} else {
			b.WriteString("  # Linux: use distro package or https://ziglang.org/download/\n")
		}
	}
	if reClang.MatchString(raw) {
		b.WriteString("  # MinGW root (MSYS2 example), if using clang fallback:\n")
		if runtime.GOOS == "windows" {
			b.WriteString("  $env:R4D_GNU_ROOT=\"C:\\msys64\\ucrt64\"\n")
		}
	}
	if rePython.MatchString(raw) {
		b.WriteString("  # PyQt5 / Python ecosystem (for .py via `r4d script.py`, not .r4d):\n")
		b.WriteString("  python -m pip install PyQt5\n")
		b.WriteString("  r4d myapp.py\n")
	}
	if strings.Contains(raw, "roma4d.toml") {
		b.WriteString("  # One-shot Windows PATH + R4D_PKG_ROOT:\n")
		b.WriteString("  .\\scripts\\Install-R4dUserEnvironment.ps1\n")
	}
	b.WriteString("--- end commands ---\n\n")
	return b.String()
}

func guideMemory() string {
	return `--- Programming Guide (what the Expert "knows") ---
Canonical doc: docs/Roma4D_Guide.md — table of contents §1–§28.
Mental model: SoA + vec4 worldtubes + par for; spacetime tokens are compile-time staging (§12).
LLM must read: §27 hard rules, §22 Python invalid patterns, §26 checklist before submitting code.
Toolchain: §1 (zig cc / clang), §19 (debug/last_build_failure.log, R4D_DEBUG).
CLI / root: §3–§4, §24 (R4D_PKG_ROOT, embedded root).
--- end guide memory ---

`
}

func llmBriefing(ctx *FailureContext, raw string, line int) string {
	var b strings.Builder
	b.WriteString("--- LLM_INSTRUCTIONS (paste into your assistant) ---\n")
	b.WriteString("You are fixing Roma4D source (.r4d), NOT CPython. Rules:\n")
	b.WriteString("1. Open docs/Roma4D_Guide.md — enforce §27 and §22.\n")
	b.WriteString("2. Do not use f-strings, import * , time(), or arbitrary PyPI imports.\n")
	b.WriteString("3. Prefer patterns from examples/hello_4d.r4d or demos/spacetime_collider.r4d.\n")
	b.WriteString("4. SoA: every read of a soa field must be followed by write-back before another read.\n")
	b.WriteString("5. After editing, user should run: r4d <file>.r4d\n")
	b.WriteString("6. If link fails, read debug/last_build_failure.log and suggest Zig on PATH or R4D_GNU_ROOT.\n\n")
	b.WriteString("Current failure summary for the model:\n")
	b.WriteString(fmt.Sprintf("- file: %s\n", ctx.SourcePath))
	b.WriteString(fmt.Sprintf("- line hint: %d\n", line))
	b.WriteString("- stderr/compiler excerpt (verbatim):\n")
	b.WriteString(raw)
	b.WriteString("\n--- END_LLM_INSTRUCTIONS ---\n")
	return b.String()
}

// suggestedPatchText returns a small illustrative fix when we have high confidence.
func suggestedPatchText(ctx *FailureContext) string {
	raw := ctx.RawError
	switch {
	case strings.Contains(raw, "import * is not supported"):
		return "# Replace: from libgeo import *\n# With:    from libgeo import bump, identity_v4\n"
	case strings.Contains(raw, "synthesized return"):
		return "def main() -> int:\n    # ... your code ...\n    return 0\n"
	case strings.Contains(raw, "UseAfterMoveError") || strings.Contains(raw, "soa field"):
		return "    col: vec4 = cell.pos\n    col = col * rot\n    cell.pos = col\n    # read cell.pos again only after assigning back\n"
	default:
		return ""
	}
}

func promptPatchAck(ctx *FailureContext) {
	_, _ = fmt.Fprintf(ctx.stderr(), "\nApply the suggested edit in your editor, save, then re-run %s. Done? [y/N]: ", ctx.Tool)
	rd := bufio.NewReader(ctx.stdin())
	line, err := rd.ReadString('\n')
	if err != nil {
		return
	}
	line = strings.TrimSpace(strings.ToLower(line))
	if line == "y" || line == "yes" {
		_, _ = fmt.Fprintf(ctx.stderr(), "Great — run `%s %s` again to verify.\n", ctx.Tool, ctx.SourcePath)
	} else {
		_, _ = fmt.Fprintf(ctx.stderr(), "Understood — keep editing, then rebuild.\n")
	}
}

func runInteractiveExpert(ctx *FailureContext) {
	_, _ = fmt.Fprintf(ctx.stderr(), "\n=== Roma4D Expert interactive (type 'help', or 'quit' to exit) ===\n")

	sc := bufio.NewScanner(ctx.stdin())
	const maxLine = 256 * 1024
	buf := make([]byte, 0, 64*1024)
	sc.Buffer(buf, maxLine)

	for {
		_, _ = fmt.Fprintf(ctx.stderr(), "expert> ")
		if !sc.Scan() {
			break
		}
		q := strings.TrimSpace(sc.Text())
		if q == "" {
			continue
		}
		low := strings.ToLower(q)
		switch low {
		case "quit", "exit", "q":
			_, _ = fmt.Fprintf(ctx.stderr(), "Goodbye.\n")
			return
		case "help", "h", "?":
			printInteractiveHelp(ctx)
		default:
			answerInteractive(ctx, low, q)
		}
	}
}

func printInteractiveHelp(ctx *FailureContext) {
	_, _ = fmt.Fprintf(ctx.stderr(), `Commands (natural language also works):
  why / reason     — what this error usually means
  zig / linker     — toolchain on Windows
  pyqt / python    — using Python vs Roma4D
  guide            — where the spec lives
  llm              — how to brief an AI assistant
  strict           — difference vs forgiving mode
  quit             — leave expert shell
`)
}

func answerInteractive(ctx *FailureContext, low, original string) {
	switch {
	case reWhy.MatchString(original):
		_, _ = fmt.Fprintf(ctx.stderr(), "This run failed during %q. Read the 'raw message' and 'symptom hint' sections above. Full spec: docs/Roma4D_Guide.md §19.\n", ctx.Verb)
	case reZig.MatchString(low):
		_, _ = fmt.Fprintf(ctx.stderr(), "Zig is the default Windows linker driver (zig cc). Install Zig, add to PATH, or set R4D_ZIG. See Guide §1 and §19.\n")
	case rePython.MatchString(low):
		_, _ = fmt.Fprintf(ctx.stderr(), "Roma4D .r4d files are AOT-compiled, not CPython. For PyQt/torch/numpy use `r4d yourapp.py` (real interpreter). HyperEngine: `r4d --explain yourapp.py` (Phase 2 Kinetic Bridge lists liftable loops + generated .r4d). If a loop is blocked, the explain output names the Python construct (e.g. user function call, dict, attribute). Guide §21.\n")
	case strings.Contains(low, "kinetic") || strings.Contains(low, "hyperengine"):
		_, _ = fmt.Fprintf(ctx.stderr(), "Kinetic Bridge: numeric `range` loops with float/int math may emit a Roma4D kernel in `r4d --explain`. Dynamic calls, attributes, subscripts, imports inside the loop block lifting. Set R4D_KINETIC_TRY_COMPILE=1 to trial-compile the generated kernel (temporal abort vs predicted CPython budget).\n")
	case strings.Contains(low, "gguf") || strings.Contains(low, "cognitive") || strings.Contains(low, "llm"):
		_, _ = fmt.Fprintf(ctx.stderr(), "Phase 3 cognitive: configure roma4d.toml [llm] model_path to a local GGUF + install llama-cpp-python. `r4d --explain` on .py runs GGUF on blocked Kinetic loops. R4D_COGNITIVE=0 disables; inference is bounded by a Z-axis (time) budget vs predicted CPython cost.\n")
	case reGuide.MatchString(low):
		_, _ = fmt.Fprintf(ctx.stderr(), "Authoritative reference: docs/Roma4D_Guide.md (repo root). Start with Mental model + §27 LLM hard rules.\n")
	case strings.Contains(low, "llm") || strings.Contains(low, "gpt") || strings.Contains(low, "claude"):
		_, _ = fmt.Fprintf(ctx.stderr(), "Copy the LLM_INSTRUCTIONS block from the debug output above into your assistant, plus the raw error. Ask it to emit only valid .r4d per Guide §25–§27.\n")
	case strings.Contains(low, "strict"):
		_, _ = fmt.Fprintf(ctx.stderr(), "Forgiving (default): rich Expert + this session. Strict: raw compiler output only — use in CI (`r4d --strict ...`).\n")
	default:
		_, _ = fmt.Fprintf(ctx.stderr(), "Try: why | zig | pyqt | guide | llm | gguf | kinetic | strict | help | quit\n")
	}
}

// readLineWindow returns a small text window around 1-based line using bufio.Scanner only.
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
