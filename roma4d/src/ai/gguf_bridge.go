// Copyright RomanAILabs - Daniel Harding
//
// GGUF Bridge — Phase 3: llama-cpp-python cognitive fallback when Kinetic static analysis blocks a loop.

package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// LLMConfig is read from roma4d.toml [llm].
type LLMConfig struct {
	ModelPath        string
	NGPULayers       int
	CognitiveEnabled bool // default true when model_path is set and file exists
}

// CognitiveSuggestion is one GGUF reasoning result for a blocked loop (telemetry + optional kernel text).
type CognitiveSuggestion struct {
	LoopLine        int           `json:"loop_line"`
	BlockReason     string        `json:"block_reason"`
	Snippet         string        `json:"snippet"`
	ConfidenceWXYZ  [4]float64    `json:"confidence_wxyz"` // W,X,Y,Z — W homogenous confidence; Z encodes time/budget alignment
	InferenceTime   time.Duration `json:"-"`
	InferenceMs     int           `json:"inference_ms,omitempty"`
	AbortedSlow     bool          `json:"aborted_slow"`
	SuggestedKernel string        `json:"suggested_kernel_r4d,omitempty"`
	Rationale       string        `json:"rationale,omitempty"`
	LiftableHint    bool          `json:"liftable_after_refactor,omitempty"`
	RawError        string        `json:"error,omitempty"`
}

// LoadLLMConfig reads [llm] from roma4d.toml next to pkgRoot (path to dir containing roma4d.toml).
func LoadLLMConfig(pkgRoot string) (LLMConfig, error) {
	var c LLMConfig
	c.NGPULayers = 0
	c.CognitiveEnabled = false
	if pkgRoot == "" {
		return c, fmt.Errorf("empty pkg root")
	}
	p := filepath.Join(pkgRoot, "roma4d.toml")
	b, err := os.ReadFile(p)
	if err != nil {
		return c, err
	}
	c = parseLLMSection(string(b), pkgRoot)
	return c, nil
}

func parseLLMSection(src, pkgRoot string) LLMConfig {
	var c LLMConfig
	var cogExplicit bool
	var cogVal bool
	section := ""
	for _, line := range strings.Split(src, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		if section != "llm" {
			continue
		}
		key, val, ok := splitTomlKeyVal(line)
		if !ok {
			continue
		}
		switch key {
		case "model_path":
			c.ModelPath = unquoteToml(val)
		case "n_gpu_layers":
			if n, err := strconv.Atoi(strings.TrimSpace(val)); err == nil {
				c.NGPULayers = n
			}
		case "cognitive_enabled":
			cogExplicit = true
			cogVal = parseTomlBoolLLM(val)
		}
	}
	if c.ModelPath != "" && !filepath.IsAbs(c.ModelPath) {
		c.ModelPath = filepath.Join(pkgRoot, filepath.Clean(c.ModelPath))
	}
	modelOK := strings.TrimSpace(c.ModelPath) != ""
	if modelOK {
		if _, err := os.Stat(c.ModelPath); err != nil {
			modelOK = false
		}
	}
	env := strings.TrimSpace(os.Getenv("R4D_COGNITIVE"))
	switch env {
	case "0", "off", "false":
		c.CognitiveEnabled = false
	case "1", "on", "true":
		c.CognitiveEnabled = modelOK
	default:
		if cogExplicit {
			c.CognitiveEnabled = cogVal && modelOK
		} else {
			c.CognitiveEnabled = modelOK
		}
	}
	return c
}

func splitTomlKeyVal(line string) (key, val string, ok bool) {
	i := strings.Index(line, "=")
	if i < 0 {
		return "", "", false
	}
	return strings.TrimSpace(line[:i]), strings.TrimSpace(line[i+1:]), true
}

func unquoteToml(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return strings.Trim(s, `"`)
	}
	return s
}

func parseTomlBoolLLM(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "true" || s == "1" || s == "yes"
}

// cognitiveModelReady returns true when GGUF path exists and [llm] + env allow cognitive.
func cognitiveModelReady(cfg LLMConfig) bool {
	if !cfg.CognitiveEnabled {
		return false
	}
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return false
	}
	_, err := os.Stat(cfg.ModelPath)
	return err == nil
}

// ExtractPythonSnippetAroundLine returns a small window of source around 1-based line number.
func ExtractPythonSnippetAroundLine(absPath string, centerLine int, halfWindow int) string {
	if centerLine < 1 || halfWindow < 1 {
		halfWindow = 6
	}
	b, err := os.ReadFile(absPath)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(b), "\n")
	start := centerLine - 1 - halfWindow
	if start < 0 {
		start = 0
	}
	end := centerLine - 1 + halfWindow
	if end > len(lines) {
		end = len(lines)
	}
	var sb strings.Builder
	for i := start; i < end; i++ {
		sb.WriteString(fmt.Sprintf("%4d | %s\n", i+1, lines[i]))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// AttachCognitiveSuggestions runs GGUF inference for each non-liftable loop (Phase 3).
// Call only when `r4d --explain` (or equivalent); respects R4D_COGNITIVE and roma4d.toml [llm].
func AttachCognitiveSuggestions(ctx context.Context, rep *KineticReport, pyPath, pyExe string, pyPrefix []string, pkgRoot string) {
	if rep == nil || pkgRoot == "" {
		return
	}
	cfg, err := LoadLLMConfig(pkgRoot)
	if err != nil {
		return
	}
	// Allow forcing cognitive attempt when model path set in toml but file missing — skip
	if !cognitiveModelReady(cfg) {
		return
	}
	script := filepath.Join(pkgRoot, "scripts", "gguf_inference.py")
	if _, err := os.Stat(script); err != nil {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}
	_ = ctx

	absPy, _ := filepath.Abs(pyPath)
	for _, L := range rep.Loops {
		if L.Liftable {
			continue
		}
		snip := ExtractPythonSnippetAroundLine(absPy, L.Lineno, 8)
		if snip == "" {
			snip = fmt.Sprintf("(line %d: snippet unavailable)", L.Lineno)
		}
		budget := predictCPythonBudget(L.RangeStop)
		budgetMs := int(budget / time.Millisecond)
		if budgetMs < 1 {
			budgetMs = 1
		}
		maxInf := cognitiveMaxInferenceBudget(budget)

		req := map[string]any{
			"snippet":               snip,
			"block_reason":          L.BlockReason,
			"predicted_budget_ms":   budgetMs,
			"loop_line":             L.Lineno,
			"range_stop_expression": L.RangeStop,
		}
		tmp, err := os.CreateTemp("", "r4d-cognitive-req-*.json")
		if err != nil {
			continue
		}
		reqPath := tmp.Name()
		_ = json.NewEncoder(tmp).Encode(req)
		_ = tmp.Close()
		defer os.Remove(reqPath)

		loopCtx, cancel := context.WithTimeout(context.Background(), maxInf)
		t0 := time.Now()
		args := append(append([]string{}, pyPrefix...), script, reqPath, cfg.ModelPath, strconv.Itoa(cfg.NGPULayers))
		cmd := exec.CommandContext(loopCtx, pyExe, args...)
		out, errRun := cmd.Output()
		cancel()
		elapsed := time.Since(t0)

		sug := CognitiveSuggestion{
			LoopLine:    L.Lineno,
			BlockReason: L.BlockReason,
			Snippet:     snip,
			InferenceTime: elapsed,
		}

		if elapsed > maxInf || loopCtx.Err() == context.DeadlineExceeded {
			sug.AbortedSlow = true
			sug.Rationale = fmt.Sprintf("temporal abort: LLM inference exceeded Z-axis budget (%v vs predicted CPython %v)", elapsed, budget)
			sug.ConfidenceWXYZ = [4]float64{0, 0, 0, 0.05} // Z≈0: time constraint violated
			rep.Cognitive = append(rep.Cognitive, sug)
			continue
		}

		if errRun != nil {
			sug.RawError = errRun.Error()
			if len(out) > 0 {
				sug.RawError += ": " + string(out)
			}
			rep.Cognitive = append(rep.Cognitive, sug)
			continue
		}

		var raw map[string]json.RawMessage
		if err := json.Unmarshal(bytesTrimLine(out), &raw); err != nil {
			sug.RawError = "json_parse: " + err.Error()
			rep.Cognitive = append(rep.Cognitive, sug)
			continue
		}
		var ok bool
		_ = json.Unmarshal(raw["ok"], &ok)
		if !ok {
			var detail string
			_ = json.Unmarshal(raw["detail"], &detail)
			var emsg string
			_ = json.Unmarshal(raw["error"], &emsg)
			sug.RawError = emsg + ": " + detail
			rep.Cognitive = append(rep.Cognitive, sug)
			continue
		}
		_ = json.Unmarshal(raw["suggested_kernel_r4d"], &sug.SuggestedKernel)
		_ = json.Unmarshal(raw["rationale"], &sug.Rationale)
		_ = json.Unmarshal(raw["liftable_after_refactor"], &sug.LiftableHint)
		var ims int
		_ = json.Unmarshal(raw["inference_ms"], &ims)
		sug.InferenceMs = ims
		var wxyz []float64
		_ = json.Unmarshal(raw["confidence_wxyz"], &wxyz)
		if len(wxyz) == 4 {
			sug.ConfidenceWXYZ = [4]float64{wxyz[0], wxyz[1], wxyz[2], wxyz[3]}
		} else {
			sug.ConfidenceWXYZ = [4]float64{0.5, 0.5, 0.5, 0.5}
		}
		// W,X,Y,Z — Z = temporal alignment (inference vs predicted CPython budget)
		if budgetMs > 0 && ims >= 0 {
			z := 1.0 - min64(1.0, float64(ims)/float64(budgetMs))
			sug.ConfidenceWXYZ[3] = z
		}
		rep.Cognitive = append(rep.Cognitive, sug)
	}
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func max64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func cognitiveMaxInferenceBudget(predictedCPython time.Duration) time.Duration {
	d := predictedCPython / 4
	if d < 3*time.Second {
		d = 3 * time.Second
	}
	if d > 45*time.Second {
		d = 45 * time.Second
	}
	return d
}
