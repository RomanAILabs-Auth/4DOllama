package compiler

import (
	"fmt"
	"os"
	"strings"
)

// Manifest is the subset of roma4d.toml needed for package identity and resolution.
type Manifest struct {
	Name        string
	Version     string
	Description string
	// Build* reflect [build] (optional).
	BuildIncremental bool // true when incremental = true under [build]
	// Systems* reflect [systems] (optional); used for codegen / toolchain flags.
	SystemsGCEnabled bool // true when gc = true; false when gc = false or omitted
	SystemsUnsafe    bool // true when unsafe = true
}

// LoadManifest reads roma4d.toml and extracts top-level name/version and [package] description.
func LoadManifest(path string) (*Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read roma4d.toml: %w", err)
	}
	m := parseTomlLite(string(b))
	if m.Name == "" {
		return nil, fmt.Errorf("roma4d.toml: missing required top-level `name = \"...\"`")
	}
	return m, nil
}

// parseTomlLite handles a tiny subset: comments, blank lines, key = "value" / key = bare,
// and [section] headers (only used to skip package subsection keys into the right field).
func parseTomlLite(src string) *Manifest {
	m := &Manifest{}
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
		key, val, ok := splitKeyVal(line)
		if !ok {
			continue
		}
		switch {
		case section == "" && key == "name":
			m.Name = unquote(val)
		case section == "" && key == "version":
			m.Version = unquote(val)
		case section == "package" && key == "description":
			m.Description = unquote(val)
		case section == "systems" && key == "gc":
			m.SystemsGCEnabled = parseTomlBool(val)
		case section == "systems" && key == "unsafe":
			m.SystemsUnsafe = parseTomlBool(val)
		case section == "build" && key == "incremental":
			m.BuildIncremental = parseTomlBool(val)
		}
	}
	return m
}

func splitKeyVal(line string) (key, val string, ok bool) {
	i := strings.Index(line, "=")
	if i < 0 {
		return "", "", false
	}
	key = strings.TrimSpace(line[:i])
	val = strings.TrimSpace(line[i+1:])
	return key, val, true
}

func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		return strings.ReplaceAll(s[1:len(s)-1], `\"`, `"`)
	}
	return s
}

func parseTomlBool(val string) bool {
	v := strings.ToLower(strings.TrimSpace(unquote(val)))
	return v == "true" || v == "1" || v == "yes"
}
