// Package ai — HyperEngine: Phase 1 AST, Phase 2 Kinetic (kinetic_bridge.go), Phase 3 GGUF cognitive (gguf_bridge.go).
//
// Design: fail-open. Default .py execution stays CPython; optional native kernel text and GGUF suggestions on --explain.
package ai

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// HyperEngineVersion is bumped when the analysis contract changes.
const HyperEngineVersion = "0.3.0-phase3"

// astDumpScript is written to a temp file and executed by the system Python.
// Output: one JSON object per line (compact) to stdout; stderr for errors.
const astDumpScript = `import ast, json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    src = f.read()
try:
    tree = ast.parse(src, filename=path)
except SyntaxError as e:
    json.dump({"ok": False, "syntax_error": str(e), "lineno": getattr(e, "lineno", 0)}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)
out = {
    "ok": True,
    "for_loops": 0,
    "while_loops": 0,
    "list_comp": 0,
    "dict_comp": 0,
    "set_comp": 0,
    "functions": 0,
    "classes": 0,
    "imports": [],
    "calls": [],  # top-level simple names seen in expressions (sample)
}
class V(ast.NodeVisitor):
    def visit_For(self, n):
        out["for_loops"] += 1
        self.generic_visit(n)
    def visit_While(self, n):
        out["while_loops"] += 1
        self.generic_visit(n)
    def visit_ListComp(self, n):
        out["list_comp"] += 1
        self.generic_visit(n)
    def visit_DictComp(self, n):
        out["dict_comp"] += 1
        self.generic_visit(n)
    def visit_SetComp(self, n):
        out["set_comp"] += 1
        self.generic_visit(n)
    def visit_FunctionDef(self, n):
        out["functions"] += 1
        self.generic_visit(n)
    def visit_AsyncFunctionDef(self, n):
        out["functions"] += 1
        self.generic_visit(n)
    def visit_ClassDef(self, n):
        out["classes"] += 1
        self.generic_visit(n)
    def visit_Import(self, n):
        for a in n.names:
            out["imports"].append(a.name.split(".")[0])
        self.generic_visit(n)
    def visit_ImportFrom(self, n):
        if n.module:
            out["imports"].append(n.module.split(".")[0])
        self.generic_visit(n)
V().visit(tree)
json.dump(out, sys.stdout)
sys.stdout.write("\n")
`

var (
	reForLine       = regexp.MustCompile(`^\s*for\s+\w+\s+in\s+`)
	reHeavyImport   = regexp.MustCompile(`(?i)^(import|from)\s+(numpy|torch|pandas|PyQt5|PyQt6|tensorflow|cv2)\b`)
	rePossibleVec   = regexp.MustCompile(`(?i)\*\s*rotor|vec4|multivector|quaternion`)
)

// RegionKind partitions the execution graph (future: native vs embedded CPython).
type RegionKind int

const (
	RegionPythonFallback RegionKind = iota
	RegionNativeCandidate
)

// Finding is one optimization or classification signal.
type Finding struct {
	Category   string  // e.g. parallel_loop, heavy_framework, comprehension
	Line       int     // 1-based; 0 if unknown
	Detail     string
	Confidence float64 // 0..1 advisory
	Kind       RegionKind
}

// ExecutionPlan is the HyperEngine output for a single .py file (Phase 1: planning only).
type ExecutionPlan struct {
	Path           string
	Version        string
	NativeHints    int
	FallbackHints  int
	Findings       []Finding
	ASTSummary     astSummary
	SyntaxOK       bool
	SyntaxErrorMsg string
	// Kinetic is filled when Phase-2 extraction runs (e.g. r4d --explain).
	Kinetic *KineticReport
}

type astSummary struct {
	ForLoops   int      `json:"for_loops"`
	WhileLoops int      `json:"while_loops"`
	ListComp   int      `json:"list_comp"`
	Functions  int      `json:"functions"`
	Imports    []string `json:"imports"`
}

type astDumpOut struct {
	OK           bool     `json:"ok"`
	SyntaxError  string   `json:"syntax_error"`
	Lineno       int      `json:"lineno"`
	ForLoops     int      `json:"for_loops"`
	WhileLoops   int      `json:"while_loops"`
	ListComp     int      `json:"list_comp"`
	Functions    int      `json:"functions"`
	Imports      []string `json:"imports"`
}

// HyperOptions tune analysis and banners.
type HyperOptions struct {
	Explain       bool
	NoBanner      bool // R4D_HYPERENGINE=0 or --no-hyperengine
	AnalysisTimeout time.Duration
}

// DefaultHyperOptions returns standard timeouts (strict guard on subprocess).
func DefaultHyperOptions() HyperOptions {
	return HyperOptions{AnalysisTimeout: 3 * time.Second}
}

// AnalyzePython inspects a Python file: AST pass (subprocess) + fast Go heuristics.
// Always returns non-nil plan on success; errors are I/O or missing interpreter (caller may ignore).
func AnalyzePython(ctx context.Context, pyPath string, pyExe string, pyPrefix []string, opt HyperOptions) (*ExecutionPlan, error) {
	plan := &ExecutionPlan{
		Path:    pyPath,
		Version: HyperEngineVersion,
	}
	if opt.AnalysisTimeout <= 0 {
		opt.AnalysisTimeout = 3 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, opt.AnalysisTimeout)
	defer cancel()

	abs, err := filepath.Abs(pyPath)
	if err != nil {
		return nil, err
	}
	plan.Path = abs

	// --- Subprocess AST (real ast module) ---
	if pyExe != "" {
		if sum, synOK, synMsg := runASTDump(ctx, pyExe, pyPrefix, abs); sum != nil {
			plan.SyntaxOK = synOK
			plan.SyntaxErrorMsg = synMsg
			plan.ASTSummary = *sum
			if synOK {
				plan.addASTFindings(sum)
			}
		}
	}

	// --- Line heuristics (allocation-free scan; bounded file read) ---
	if err := scanLinesHeuristic(abs, plan); err != nil && !errors.Is(err, io.EOF) {
		// non-fatal
		_ = err
	}

	plan.recomputeCounts()
	return plan, nil
}

func runASTDump(ctx context.Context, pyExe string, pyPrefix []string, scriptPath string) (*astSummary, bool, string) {
	tmp, err := os.CreateTemp("", "r4d-hyper-ast-*.py")
	if err != nil {
		return nil, true, ""
	}
	path := tmp.Name()
	_, _ = tmp.WriteString(astDumpScript)
	_ = tmp.Close()
	defer os.Remove(path)

	args := append(append([]string{}, pyPrefix...), path, scriptPath)
	cmd := exec.CommandContext(ctx, pyExe, args...)
	out, err := cmd.Output()
	if err != nil {
		return nil, true, ""
	}
	line := strings.TrimSpace(string(out))
	var dump astDumpOut
	if err := json.Unmarshal([]byte(line), &dump); err != nil {
		return nil, true, ""
	}
	if !dump.OK {
		return nil, false, dump.SyntaxError
	}
	s := astSummary{
		ForLoops:   dump.ForLoops,
		WhileLoops: dump.WhileLoops,
		ListComp:   dump.ListComp,
		Functions:  dump.Functions,
		Imports:    dump.Imports,
	}
	return &s, true, ""
}

func (p *ExecutionPlan) addASTFindings(s *astSummary) {
	if s.ForLoops > 0 {
		p.Findings = append(p.Findings, Finding{
			Category:   "parallel_loop_candidate",
			Line:       0,
			Detail:     fmt.Sprintf("AST: %d for-loop(s) — if body is side-effect-free over SoA data, candidate for Roma4D par for", s.ForLoops),
			Confidence: 0.35,
			Kind:       RegionNativeCandidate,
		})
	}
	if s.ListComp > 0 {
		p.Findings = append(p.Findings, Finding{
			Category:   "comprehension_batch",
			Line:       0,
			Detail:     fmt.Sprintf("AST: %d list comprehension(s) — may map to vectorized / SoA kernels when dependencies are pure", s.ListComp),
			Confidence: 0.25,
			Kind:       RegionNativeCandidate,
		})
	}
	for _, imp := range s.Imports {
		low := strings.ToLower(imp)
		switch low {
		case "numpy", "torch", "pandas", "tensorflow", "cv2":
			p.Findings = append(p.Findings, Finding{
				Category:   "heavy_framework",
				Line:       0,
				Detail:     fmt.Sprintf("import graph includes %q — stays on CPython until a safe tensor bridge exists", imp),
				Confidence: 1.0,
				Kind:       RegionPythonFallback,
			})
		case "pyqt5", "pyqt6", "pyside2", "pyside6":
			p.Findings = append(p.Findings, Finding{
				Category:   "ui_runtime",
				Line:       0,
				Detail:     fmt.Sprintf("UI stack (%s) — HyperEngine keeps event loop on CPython (mandatory fallback)", imp),
				Confidence: 1.0,
				Kind:       RegionPythonFallback,
			})
		}
	}
}

func scanLinesHeuristic(abs string, p *ExecutionPlan) error {
	f, err := os.Open(abs)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	const maxLines = 100_000
	lineNo := 0
	for sc.Scan() {
		lineNo++
		if lineNo > maxLines {
			break
		}
		line := sc.Text()
		if reForLine.MatchString(line) {
			p.Findings = append(p.Findings, Finding{
				Category:   "text_for_loop",
				Line:       lineNo,
				Detail:     "Heuristic: `for … in …` line — review for Roma4D par for if iterable is columnar / SIMD-friendly",
				Confidence: 0.2,
				Kind:       RegionNativeCandidate,
			})
		}
		if reHeavyImport.MatchString(line) {
			p.Findings = append(p.Findings, Finding{
				Category:   "heavy_import_line",
				Line:       lineNo,
				Detail:     "Heuristic: scientific/UI import — default execution region: CPython",
				Confidence: 0.9,
				Kind:       RegionPythonFallback,
			})
		}
		if rePossibleVec.MatchString(line) {
			p.Findings = append(p.Findings, Finding{
				Category:   "geometric_hint",
				Line:       lineNo,
				Detail:     "Heuristic: geometric token — if logic is pure, strong candidate for vec4/rotor lowering in Roma4D",
				Confidence: 0.4,
				Kind:       RegionNativeCandidate,
			})
		}
	}
	return sc.Err()
}

func (p *ExecutionPlan) recomputeCounts() {
	p.NativeHints = 0
	p.FallbackHints = 0
	for _, f := range p.Findings {
		switch f.Kind {
		case RegionNativeCandidate:
			p.NativeHints++
		case RegionPythonFallback:
			p.FallbackHints++
		}
	}
}

// PrintBanner prints the user-facing HyperEngine line (stderr).
func PrintBanner(w io.Writer, tool string, plan *ExecutionPlan) {
	if plan == nil {
		fmt.Fprintf(w, "[ Roma4D HyperEngine %s ] Running Python (analysis unavailable)\n", HyperEngineVersion)
		return
	}
	fmt.Fprintf(w, "[ Roma4D HyperEngine %s ] Running Python with 4D acceleration roadmap…\n", HyperEngineVersion)
	if plan.SyntaxOK {
		fmt.Fprintf(w, "  └─ analysis: %d native-oriented hint(s), %d CPython fallback signal(s) (advisory only; execution = standard interpreter)\n",
			plan.NativeHints, plan.FallbackHints)
	} else if plan.SyntaxErrorMsg != "" {
		fmt.Fprintf(w, "  └─ AST parse: syntax error — %s\n", plan.SyntaxErrorMsg)
	}
}

// Explain writes a human-readable report (for --explain).
func Explain(w io.Writer, tool string, plan *ExecutionPlan) {
	if plan == nil {
		fmt.Fprintf(w, "%s: HyperEngine: no plan\n", tool)
		return
	}
	fmt.Fprintf(w, "=== Roma4D HyperEngine explain (%s) ===\n", HyperEngineVersion)
	fmt.Fprintf(w, "File: %s\n", plan.Path)
	fmt.Fprintf(w, "Syntax OK: %v\n", plan.SyntaxOK)
	if !plan.SyntaxOK && plan.SyntaxErrorMsg != "" {
		fmt.Fprintf(w, "Parse error: %s\n", plan.SyntaxErrorMsg)
	}
	fmt.Fprintf(w, "AST snapshot: for=%d while=%d list_comp=%d funcs=%d imports=%v\n",
		plan.ASTSummary.ForLoops, plan.ASTSummary.WhileLoops, plan.ASTSummary.ListComp,
		plan.ASTSummary.Functions, plan.ASTSummary.Imports)
	fmt.Fprintf(w, "Findings (%d):\n", len(plan.Findings))
	for i, f := range plan.Findings {
		k := "PYTHON"
		if f.Kind == RegionNativeCandidate {
			k = "NATIVE?"
		}
		ln := ""
		if f.Line > 0 {
			ln = fmt.Sprintf(" line %d —", f.Line)
		}
		fmt.Fprintf(w, "  %d. [%s]%s %s (conf=%.2f)\n     %s\n", i+1, k, ln, f.Category, f.Confidence, f.Detail)
	}
	if plan.Kinetic != nil {
		fmt.Fprintf(w, "\n--- Kinetic Bridge (Phase 2) ---\n")
		if len(plan.Kinetic.Loops) == 0 && plan.Kinetic.CompileLog != "" {
			fmt.Fprintf(w, "Extraction note: %s\n", plan.Kinetic.CompileLog)
		}
		for _, L := range plan.Kinetic.Loops {
			if L.Liftable {
				fmt.Fprintf(w, "  line %d: LIFTABLE (loop var %q, accumulator %q)\n", L.Lineno, L.Target, L.AccVar)
			} else {
				ex := FormatKineticExpertBlocked(L)
				if ex != "" {
					fmt.Fprintf(w, "%s", ex)
				} else {
					fmt.Fprintf(w, "  line %d: not liftable — %s\n", L.Lineno, L.BlockReason)
				}
			}
		}
		if strings.TrimSpace(plan.Kinetic.GeneratedR4D) != "" {
			fmt.Fprintf(w, "\nGenerated Roma4D kernel (first liftable loop):\n")
			fmt.Fprintf(w, "%s\n", strings.TrimRight(plan.Kinetic.GeneratedR4D, "\n"))
		}
		if plan.Kinetic.CompileOK {
			fmt.Fprintf(w, "\nKinetic compile: ok (trial run %v", plan.Kinetic.RunElapsed)
			if plan.Kinetic.TemporalAbort {
				fmt.Fprintf(w, "; temporal abort — exceeded predicted CPython budget %v", plan.Kinetic.PredictedCPMax)
			}
			fmt.Fprintf(w, ")\n")
		} else if plan.Kinetic.CompileLog != "" {
			fmt.Fprintf(w, "\nKinetic compile/run: %s\n", plan.Kinetic.CompileLog)
		}
		fmt.Fprintf(w, "--- end Kinetic ---\n")
	}
	if plan.Kinetic != nil && len(plan.Kinetic.Cognitive) > 0 {
		for _, cg := range plan.Kinetic.Cognitive {
			fmt.Fprintf(w, "\n[4D-AI-SUGGESTION] loop_line=%d block_reason=%q\n", cg.LoopLine, cg.BlockReason)
			fmt.Fprintf(w, "  confidence_WXYZ (W,X,Y,Z; Z=temporal vs CPython budget): [%.3f, %.3f, %.3f, %.3f]\n",
				cg.ConfidenceWXYZ[0], cg.ConfidenceWXYZ[1], cg.ConfidenceWXYZ[2], cg.ConfidenceWXYZ[3])
			fmt.Fprintf(w, "  inference_wall=%v inference_report_ms=%d aborted_slow=%v\n",
				cg.InferenceTime, cg.InferenceMs, cg.AbortedSlow)
			if cg.AbortedSlow {
				fmt.Fprintf(w, "  %s\n", cg.Rationale)
				continue
			}
			if cg.RawError != "" {
				fmt.Fprintf(w, "  cognitive_error: %s\n", cg.RawError)
				continue
			}
			if cg.LiftableHint {
				fmt.Fprintf(w, "  liftable_after_refactor (model hint): true\n")
			}
			if strings.TrimSpace(cg.Rationale) != "" {
				fmt.Fprintf(w, "  rationale: %s\n", strings.TrimSpace(cg.Rationale))
			}
			if strings.TrimSpace(cg.SuggestedKernel) != "" {
				fmt.Fprintf(w, "  AI-suggested 4D kernel (verify before use):\n%s\n", strings.TrimRight(cg.SuggestedKernel, "\n"))
			}
		}
	}
	fmt.Fprintf(w, "Execution graph: default path remains CPython (100%%). Native kernel compile is opt-in (R4D_KINETIC_TRY_COMPILE=1).\n")
	fmt.Fprintf(w, "=== end explain ===\n")
}

// SuggestExpertHook returns lines to append to Expert / LLM context when Python perf is poor (stub for agent integration).
func SuggestExpertHook(plan *ExecutionPlan) []string {
	if plan == nil || !plan.SyntaxOK {
		return nil
	}
	var s []string
	if plan.NativeHints > 0 {
		s = append(s, fmt.Sprintf("HyperEngine: %d loop/comprehension hotspot(s) — consider rewriting hot path in .r4d with par for + SoA.", plan.NativeHints))
	}
	if plan.FallbackHints > 0 {
		s = append(s, "HyperEngine: framework/UI imports detected — keep this path in Python or isolate a numeric kernel in .r4d.")
	}
	return s
}
