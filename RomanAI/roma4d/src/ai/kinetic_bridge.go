// Copyright RomanAILabs - Daniel Harding
//
// Kinetic Bridge — Phase 2: hot-path extraction, Python→Roma4D kernel sketch, aligned buffers,
// optional compile + timed native run (temporal abort vs predicted CPython budget).

package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

// kineticExtractScript runs in CPython; emits JSON { "loops": [ ... ] }.
const kineticExtractScript = `import ast, json, sys

if not hasattr(ast, "unparse"):
    json.dump({"ok": False, "error": "Kinetic extract requires Python 3.9+ (ast.unparse)"}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)

path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    src = f.read()
try:
    tree = ast.parse(src, filename=path)
except SyntaxError as e:
    json.dump({"ok": False, "error": str(e)}, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)

FORBIDDEN_NODES = (ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await,
                   ast.Global, ast.Nonlocal, ast.ClassDef, ast.With, ast.AsyncWith,
                   ast.Try, ast.Raise, ast.Assert, ast.Import, ast.ImportFrom)

def iter_range_info(for_node):
    if not isinstance(for_node.iter, ast.Call):
        return None, "iter_not_call"
    fn = for_node.iter.func
    if not isinstance(fn, ast.Name) or fn.id != "range":
        return None, "iter_not_range"
    args = for_node.iter.args
    if len(args) == 1:
        return ("0", ast.unparse(args[0]), "1"), None
    if len(args) == 2:
        return (ast.unparse(args[0]), ast.unparse(args[1]), "1"), None
    if len(args) == 3:
        return (ast.unparse(args[0]), ast.unparse(args[1]), ast.unparse(args[2])), None
    return None, "range_arity"

def stmt_blocked(stmt):
    for n in ast.walk(stmt):
        if isinstance(n, ast.Dict):
            return "forbidden_ast:Dict"
        if isinstance(n, ast.Set):
            return "forbidden_ast:Set"
    for n in ast.walk(stmt):
        if isinstance(n, FORBIDDEN_NODES):
            return "forbidden_ast:" + type(n).__name__
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id in ("float", "int", "abs", "min", "max"):
                    continue
                return "dynamic_call:name:" + n.func.id
            return "dynamic_call:complex_callee"
        if isinstance(n, ast.Attribute):
            return "dynamic_attribute:" + ast.unparse(n)
        if isinstance(n, ast.Subscript):
            return "subscript/dynamic_index"
        if isinstance(n, ast.NamedExpr):
            return "walrus"
    return None

def classify_body(for_node):
    body = for_node.body
    if len(body) != 1:
        return None, "multi_stmt_body(%d)" % len(body)
    st = body[0]
    b = stmt_blocked(st)
    if b:
        return None, b
    return st, None

def py_expr_to_r4d(node, loop_var):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            s = repr(float(node.value)) if isinstance(node.value, float) else str(int(node.value))
            return s, None
        return None, "non_numeric_constant"
    if isinstance(node, ast.Name):
        if node.id == loop_var:
            return "float(" + loop_var + ")", None
        return None, "unknown_name:" + node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner, err = py_expr_to_r4d(node.operand, loop_var)
        if err:
            return None, err
        return "(-(" + inner + "))", None
    if isinstance(node, ast.BinOp):
        opmap = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}
        t = type(node.op)
        if t not in opmap:
            return None, "unsupported_binop"
        la, e1 = py_expr_to_r4d(node.left, loop_var)
        if e1:
            return None, e1
        ra, e2 = py_expr_to_r4d(node.right, loop_var)
        if e2:
            return None, e2
        return "(" + la + " " + opmap[t] + " " + ra + ")", None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
        if len(node.args) != 1:
            return None, "float_arity"
        a0 = node.args[0]
        if isinstance(a0, ast.Name) and a0.id == loop_var:
            return "float(" + loop_var + ")", None
        return py_expr_to_r4d(a0, loop_var)
    return None, "unsupported_expr:" + type(node).__name__

def aug_assign_r4d(stmt, loop_var):
    if not isinstance(stmt, ast.AugAssign):
        return None, "not_augassign"
    if not isinstance(stmt.target, ast.Name):
        return None, "compound_target"
    acc = stmt.target.id
    opmap = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    if type(stmt.op) not in opmap:
        return None, "unsupported_aug_op"
    rhs, err = py_expr_to_r4d(stmt.value, loop_var)
    if err:
        return None, err
    op = opmap[type(stmt.op)]
    return acc + " = " + acc + " " + op + " " + rhs, acc

def assign_accum(stmt, loop_var):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        t = stmt.targets[0].id
        if isinstance(stmt.value, ast.BinOp) and isinstance(stmt.value.op, ast.Add):
            if isinstance(stmt.value.left, ast.Name) and stmt.value.left.id == t:
                rhs, err = py_expr_to_r4d(stmt.value.right, loop_var)
                if err:
                    return None, None, err
                return t + " = " + t + " + " + rhs, t, None
    return None, None, "not_simple_assign_accum"

loops_out = []
class V(ast.NodeVisitor):
    def visit_For(self, node):
        tgt = node.target
        if not isinstance(tgt, ast.Name):
            loops_out.append({"lineno": getattr(node, "lineno", 0), "liftable": False,
                "block_reason": "non_name_target", "target": None, "range_start": None,
                "range_stop": None, "range_step": None, "acc_var": None, "r4d_rhs": None})
            self.generic_visit(node)
            return
        loop_var = tgt.id
        rng, why = iter_range_info(node)
        if rng is None:
            loops_out.append({"lineno": node.lineno, "liftable": False, "block_reason": why,
                "target": loop_var, "range_start": None, "range_stop": None, "range_step": None,
                "acc_var": None, "r4d_rhs": None})
            self.generic_visit(node)
            return
        r0, r1, rstep = rng
        st, why2 = classify_body(node)
        if st is None:
            loops_out.append({"lineno": node.lineno, "liftable": False, "block_reason": why2,
                "target": loop_var, "range_start": r0, "range_stop": r1, "range_step": rstep,
                "acc_var": None, "r4d_rhs": None})
            self.generic_visit(node)
            return
        line, acc = aug_assign_r4d(st, loop_var)
        if line is None:
            line2, acc2, why3 = assign_accum(st, loop_var)
            if line2 is None:
                loops_out.append({"lineno": node.lineno, "liftable": False,
                    "block_reason": why3 or "assign_shape", "target": loop_var,
                    "range_start": r0, "range_stop": r1, "range_step": rstep,
                    "acc_var": None, "r4d_rhs": None})
                self.generic_visit(node)
                return
            line, acc = line2, acc2
        loops_out.append({"lineno": node.lineno, "liftable": True, "block_reason": "",
            "target": loop_var, "range_start": r0, "range_stop": r1, "range_step": rstep,
            "acc_var": acc, "r4d_rhs": line})
        self.generic_visit(node)

V().visit(tree)
json.dump({"ok": True, "loops": loops_out}, sys.stdout)
sys.stdout.write("\n")
`

// KineticReport is the Phase-2 artifact for one .py file (extraction + optional compile/run).
type KineticReport struct {
	Loops          []LiftLoop              `json:"loops"`
	GeneratedR4D   string                  // full module source for first liftable loop (demo kernel)
	CompileOK      bool
	CompileLog     string
	RunElapsed     time.Duration
	TemporalAbort  bool
	PredictedCPMax time.Duration // budget used for abort decision
	Cognitive      []CognitiveSuggestion   // Phase 3: GGUF suggestions for blocked loops
}

// LiftLoop is one Python for-loop classification + optional R4D lowering line.
type LiftLoop struct {
	Lineno      int    `json:"lineno"`
	Liftable    bool   `json:"liftable"`
	BlockReason string `json:"block_reason"`
	Target      string `json:"target"`
	RangeStart  string `json:"range_start"`
	RangeStop   string `json:"range_stop"`
	RangeStep   string `json:"range_step"`
	AccVar      string `json:"acc_var"`
	R4dRhsLine  string `json:"r4d_rhs"` // full assignment including acc
}

type kineticDump struct {
	OK    bool         `json:"ok"`
	Error string       `json:"error"`
	Loops []LiftLoop   `json:"loops"`
}

// DefaultKineticTimeouts bounds subprocess and native trial (Z-axis temporal guard).
const (
	DefaultKineticExtractTimeout = 4 * time.Second
	DefaultKineticCompileTimeout = 120 * time.Second
	DefaultKineticRunTimeout     = 30 * time.Second
)

// RunKineticPipeline extracts loops, generates .r4d for the first liftable loop, optionally compiles and runs.
// pkgRoot: Roma4D tree with roma4d.toml (empty → skip compile).
// tryNative: if true and pkgRoot set, compile temp .r4d and run with temporal abort.
// FindRoma4dPackageRoot walks upward from a file or directory for roma4d.toml (same rules as CLI).
func FindRoma4dPackageRoot(from string) (string, error) {
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
		if _, err := os.Stat(filepath.Join(root, "roma4d.toml")); err == nil {
			return root, nil
		}
	}
	return "", fmt.Errorf("could not find roma4d.toml (set R4D_PKG_ROOT to your Roma4D checkout)")
}

// RunKineticPipeline extracts loops, generates .r4d for the first liftable loop, optionally compiles and runs.
// tryNative: compile+run temp kernel (use R4D_KINETIC_TRY_COMPILE=1 from CLI); temporal abort vs predicted CPython budget.
// runCognitive: when true (e.g. r4d --explain), invoke GGUF bridge for blocked loops if [llm] is configured.
func RunKineticPipeline(ctx context.Context, pyPath, pyExe string, pyPrefix []string, pkgRoot string, tryNative, runCognitive bool) (*KineticReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, DefaultKineticExtractTimeout)
	defer cancel()

	abs, err := filepath.Abs(pyPath)
	if err != nil {
		return nil, err
	}
	loops, err := runKineticExtractScript(ctx, pyExe, pyPrefix, abs)
	if err != nil {
		return &KineticReport{Loops: nil, GeneratedR4D: "", CompileLog: err.Error()}, nil
	}
	rep := &KineticReport{Loops: loops}
	for _, L := range loops {
		if L.Liftable && L.AccVar != "" && L.R4dRhsLine != "" {
			src, genErr := buildKineticModuleSource(L)
			if genErr != nil {
				rep.GeneratedR4D = fmt.Sprintf("# (internal) kernel generation error: %v\n", genErr)
				break
			}
			rep.GeneratedR4D = src
			rep.PredictedCPMax = predictCPythonBudget(L.RangeStop)
			if tryNative && pkgRoot != "" && rep.GeneratedR4D != "" {
				_ = compileAndRunKinetic(ctx, pkgRoot, rep)
			}
			break
		}
	}
	if runCognitive && pkgRoot != "" {
		AttachCognitiveSuggestions(ctx, rep, abs, pyExe, pyPrefix, pkgRoot)
	}
	return rep, nil
}

func predictCPythonBudget(rangeStopExpr string) time.Duration {
	// Heuristic O(n) budget for scalar float loop in CPython (very loose; abort protects native trial).
	n := parseIntish(rangeStopExpr)
	if n <= 0 {
		n = 1_000_000
	}
	ms := 50 + n/100_000
	if ms > 8000 {
		ms = 8000
	}
	return time.Duration(ms) * time.Millisecond
}

func parseIntish(s string) int64 {
	s = strings.ReplaceAll(strings.TrimSpace(s), "_", "")
	var n int64
	for _, c := range s {
		if c < '0' || c > '9' {
			return 1_000_000
		}
		n = n*10 + int64(c-'0')
		if n > 1<<62 {
			return 1 << 62
		}
	}
	return n
}

func runKineticExtractScript(ctx context.Context, pyExe string, pyPrefix []string, scriptPath string) ([]LiftLoop, error) {
	tmp, err := os.CreateTemp("", "r4d-kinetic-*.py")
	if err != nil {
		return nil, err
	}
	p := tmp.Name()
	_, _ = tmp.WriteString(kineticExtractScript)
	_ = tmp.Close()
	defer os.Remove(p)

	args := append(append([]string{}, pyPrefix...), p, scriptPath)
	cmd := exec.CommandContext(ctx, pyExe, args...)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("kinetic extract: %w", err)
	}
	var d kineticDump
	if err := json.Unmarshal(bytesTrimLine(out), &d); err != nil {
		return nil, fmt.Errorf("kinetic extract json: %w", err)
	}
	if !d.OK {
		return nil, fmt.Errorf("kinetic extract: %s", d.Error)
	}
	return d.Loops, nil
}

func bytesTrimLine(b []byte) []byte {
	b = []byte(strings.TrimSpace(string(b)))
	return b
}

func buildKineticModuleSource(L LiftLoop) (string, error) {
	// Serial `for` — safe for accumulators (no bogus parallel reduction).
	// Range: Roma4D accepts range(stop) with named stop as int.
	stop := strings.TrimSpace(L.RangeStop)
	if stop == "" {
		return "", fmt.Errorf("empty range stop")
	}
	acc := L.AccVar
	lv := L.Target
	if acc == "" || lv == "" {
		return "", fmt.Errorf("missing acc or loop var")
	}
	bodyLine := strings.TrimSpace(L.R4dRhsLine)
	// Map Python // to Roma4D if needed (keep as-is for now)
	hdr := `# Kinetic Bridge — auto-generated Roma4D kernel (Phase 2). Do not edit by hand.
# Source loop lifted from Python with serial semantics (accumulator-safe).

`
	mod := fmt.Sprintf("%sdef main() -> int:\n    n: int = %s\n    %s: float = 0.0\n    for %s in range(n):\n        %s\n    print(%s)\n    return 0\n",
		hdr, stop, acc, lv, bodyLine, acc)
	return mod, nil
}

// MapPythonBinOpToGA documents operator mapping for future vec4/rotor lowering (translation table).
// Today numeric kernels use scalar float in generated R4D; GA products apply when types are vec4/rotor.
func MapPythonBinOpToGA(pyOp string) (roma4dForm string, ok bool) {
	switch pyOp {
	case "*":
		return "geometric_product_or_scalar_mul", true
	case "+":
		return "add", true
	case "-":
		return "sub", true
	case "/":
		return "scalar_div", true
	case "@":
		return "inner_product_or_temporal_projection", true
	default:
		return "", false
	}
}

// AlignedFloat64 returns a slice of length n backed by 16-byte aligned memory (AVX-friendly SoA lane).
func AlignedFloat64(n int) []float64 {
	if n <= 0 {
		return nil
	}
	need := n*8 + 16
	raw := make([]byte, need)
	p := uintptr(unsafe.Pointer(&raw[0]))
	off := (16 - int(p%16)) % 16
	ptr := unsafe.Pointer(&raw[off])
	return unsafe.Slice((*float64)(ptr), n)
}

// SoA4xN stores n logical vec4s as four aligned planes (x,y,z,w). Length of each plane is n.
type SoA4xN struct {
	X, Y, Z, W []float64
	N          int
}

// NewSoA4xN allocates four 16-byte aligned planes of length n.
func NewSoA4xN(n int) SoA4xN {
	return SoA4xN{
		X: AlignedFloat64(n),
		Y: AlignedFloat64(n),
		Z: AlignedFloat64(n),
		W: AlignedFloat64(n),
		N: n,
	}
}

// BasePointer returns address of plane X[0] for FFI (read-only hint; keep slice alive).
func (s *SoA4xN) BasePointer() uintptr {
	if s.N == 0 || len(s.X) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.X[0]))
}

func compileAndRunKinetic(_ context.Context, pkgRoot string, rep *KineticReport) error {
	tmpDir, err := os.MkdirTemp("", "r4d-kinetic-build-*")
	if err != nil {
		rep.CompileLog = err.Error()
		return err
	}
	defer os.RemoveAll(tmpDir)

	srcPath := filepath.Join(tmpDir, "kinetic_lift.r4d")
	if err := os.WriteFile(srcPath, []byte(rep.GeneratedR4D), 0o644); err != nil {
		rep.CompileLog = err.Error()
		return err
	}
	exe := filepath.Join(tmpDir, "kinetic_out")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	_, err = compiler.BuildExecutable(pkgRoot, srcPath, exe, nil)
	if err != nil {
		rep.CompileLog = err.Error()
		return err
	}
	rep.CompileOK = true

	runBudget := rep.PredictedCPMax
	if runBudget < 50*time.Millisecond {
		runBudget = 50 * time.Millisecond
	}
	rctx, rcancel := context.WithTimeout(context.Background(), runBudget)
	defer rcancel()
	t0 := time.Now()
	cmd := exec.CommandContext(rctx, exe)
	cmd.Stdout = nil
	cmd.Stderr = nil
	errRun := cmd.Run()
	rep.RunElapsed = time.Since(t0)
	if rctx.Err() == context.DeadlineExceeded || errorsIsDeadline(errRun) {
		rep.TemporalAbort = true
		rep.CompileLog += "; temporal_abort: native run exceeded predicted CPython budget"
		return nil
	}
	if errRun != nil {
		rep.CompileLog += "; run: " + errRun.Error()
	}
	return nil
}

func errorsIsDeadline(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(strings.ToLower(err.Error()), "deadline") || strings.Contains(err.Error(), "context deadline exceeded")
}

// FormatKineticExpertBlocked returns copy-paste lines for Expert / CLI when lift fails.
func FormatKineticExpertBlocked(loop LiftLoop) string {
	if loop.Liftable {
		return ""
	}
	reason := loop.BlockReason
	msg := "Kinetic Bridge blocked 4D acceleration for a `for` loop"
	if loop.Lineno > 0 {
		msg += fmt.Sprintf(" at line %d", loop.Lineno)
	}
	msg += ".\n"
	switch {
	case strings.HasPrefix(reason, "dynamic_call:name:"):
		fn := strings.TrimPrefix(reason, "dynamic_call:name:")
		msg += fmt.Sprintf("Cause: call to Python function %q — dynamic dispatch not lifted (keep in CPython or rewrite as pure float/int math in-loop).\n", fn)
	case strings.HasPrefix(reason, "forbidden_ast:"):
		msg += fmt.Sprintf("Cause: unsupported AST in loop body (%s) — dict/set/class/import/with/try/async patterns stay on CPython.\n", strings.TrimPrefix(reason, "forbidden_ast:"))
	case strings.HasPrefix(reason, "dynamic_attribute:"):
		msg += "Cause: attribute access on objects — not a flat numeric kernel; use plain names and builtins float/int only.\n"
	case reason == "subscript/dynamic_index":
		msg += "Cause: subscripting — SoA bridge not proven safe for this access pattern yet.\n"
	case strings.HasPrefix(reason, "iter_not"):
		msg += "Cause: iterator is not a simple `range(...)` — Kinetic v1 only lifts range loops.\n"
	default:
		msg += "Cause: " + reason + ".\n"
	}
	return msg
}
