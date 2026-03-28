package ai4d

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// CmdKind is a Roma4D control primitive dispatched to RQ4D (no direct numerics in the parser).
type CmdKind uint8

const (
	CmdNop CmdKind = iota
	CmdModelLoad
	CmdModelSave
	CmdLayerClifford
	CmdRotate
	CmdTrain
	CmdInfer
	CmdEvolve
)

// Command is one parsed line from a .r4d AI control script.
type Command struct {
	Kind   CmdKind
	Path   string
	Size   int
	Plane  string
	Angle  float64
	Steps  int
	LR     float64
	Field  string
	Input  []float64
}

// ParseScript splits on newlines and parses each non-empty, non-comment line.
func ParseScript(src string) ([]Command, error) {
	var out []Command
	for _, line := range strings.Split(src, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		// Keep #pragma for ParseLine (core mode hints); skip other # comments (e.g. copyright).
		if strings.HasPrefix(line, "#") {
			after := strings.TrimSpace(strings.TrimPrefix(line, "#"))
			if len(after) < 6 || !strings.EqualFold(after[:6], "pragma") {
				continue
			}
		}
		c, err := ParseLine(line)
		if err != nil {
			return nil, fmt.Errorf("ai4d script: %w", err)
		}
		if c.Kind != CmdNop {
			out = append(out, c)
		}
	}
	return out, nil
}

// ParseLine parses one Roma4D-style AI primitive (case-insensitive keyword).
//
// Supported forms:
//
//	MODEL LOAD "path"
//	MODEL SAVE "path"
//	LAYER CLIFFORD size=128
//	ROTATE plane=XY angle=0.785
//	TRAIN steps=100 lr=0.001
//	INFER input=[1,0,0,0]
//	EVOLVE field=quantum steps=10
func ParseLine(line string) (Command, error) {
	parts := fieldsKeepQuoted(line)
	if len(parts) == 0 {
		return Command{}, nil
	}
	kw := strings.ToUpper(parts[0])
	if strings.EqualFold(parts[0], "#pragma") {
		return Command{}, nil
	}
	switch kw {
	case "MODEL":
		if len(parts) < 3 {
			return Command{}, fmt.Errorf("MODEL: need LOAD|SAVE and path")
		}
		op := strings.ToUpper(parts[1])
		path := unquote(parts[2])
		switch op {
		case "LOAD":
			return Command{Kind: CmdModelLoad, Path: path}, nil
		case "SAVE":
			return Command{Kind: CmdModelSave, Path: path}, nil
		default:
			return Command{}, fmt.Errorf("MODEL: unknown op %q", parts[1])
		}
	case "LAYER":
		if len(parts) < 3 || strings.ToUpper(parts[1]) != "CLIFFORD" {
			return Command{}, fmt.Errorf("LAYER: use LAYER CLIFFORD size=N")
		}
		m := parseKV(parts[2:])
		sz, err := intField(m, "size", 0)
		if err != nil {
			return Command{}, err
		}
		return Command{Kind: CmdLayerClifford, Size: sz}, nil
	case "ROTATE":
		m := parseKV(parts[1:])
		pl := strings.ToUpper(m["plane"])
		ang, err := floatField(m, "angle", 0)
		if err != nil {
			return Command{}, err
		}
		return Command{Kind: CmdRotate, Plane: pl, Angle: ang}, nil
	case "TRAIN":
		m := parseKV(parts[1:])
		st, err := intField(m, "steps", 1)
		if err != nil {
			return Command{}, err
		}
		lr, err := floatField(m, "lr", 0.001)
		if err != nil {
			return Command{}, err
		}
		return Command{Kind: CmdTrain, Steps: st, LR: lr}, nil
	case "INFER":
		m := parseKV(parts[1:])
		raw, ok := m["input"]
		if !ok {
			return Command{}, fmt.Errorf("INFER: need input=[...]")
		}
		vec, err := parseFloatVec(raw)
		if err != nil {
			return Command{}, err
		}
		return Command{Kind: CmdInfer, Input: vec}, nil
	case "EVOLVE":
		m := parseKV(parts[1:])
		f := strings.ToLower(m["field"])
		st, err := intField(m, "steps", 1)
		if err != nil {
			return Command{}, err
		}
		return Command{Kind: CmdEvolve, Field: f, Steps: st}, nil
	default:
		return Command{}, fmt.Errorf("unknown keyword %q", parts[0])
	}
}

func fieldsKeepQuoted(s string) []string {
	var out []string
	var b strings.Builder
	inQuote := false
	flush := func() {
		t := strings.TrimSpace(b.String())
		if t != "" {
			out = append(out, t)
		}
		b.Reset()
	}
	for _, r := range s {
		switch {
		case r == '"' && !inQuote:
			inQuote = true
		case r == '"' && inQuote:
			inQuote = false
		case (r == ' ' || r == '\t') && !inQuote:
			flush()
		default:
			b.WriteRune(r)
		}
	}
	flush()
	return out
}

func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return s[1 : len(s)-1]
	}
	return s
}

// parseKV parses key=value tokens; values may include = inside brackets.
func parseKV(parts []string) map[string]string {
	m := make(map[string]string)
	for _, p := range parts {
		i := strings.IndexByte(p, '=')
		if i <= 0 {
			continue
		}
		k := strings.ToLower(strings.TrimSpace(p[:i]))
		v := strings.TrimSpace(p[i+1:])
		m[k] = v
	}
	return m
}

func intField(m map[string]string, key string, def int) (int, error) {
	s, ok := m[key]
	if !ok {
		return def, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("%s: invalid int %q", key, s)
	}
	return n, nil
}

func floatField(m map[string]string, key string, def float64) (float64, error) {
	s, ok := m[key]
	if !ok {
		return def, nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: invalid float %q", key, s)
	}
	return f, nil
}

func parseFloatVec(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		return nil, fmt.Errorf("input must be [..] array")
	}
	s = s[1 : len(s)-1]
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var arr []float64
	if err := json.Unmarshal([]byte("["+s+"]"), &arr); err == nil {
		return arr, nil
	}
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		f, err := strconv.ParseFloat(tok, 64)
		if err != nil {
			return nil, err
		}
		arr = append(arr, f)
	}
	return arr, nil
}
