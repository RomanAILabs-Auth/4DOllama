package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Modelfile captures Ollama-style directives we need for 4dollama create (FROM, PARAMETER, optional TEMPLATE).
type Modelfile struct {
	FromPaths      []string
	TokenizerFrom  string // optional GGUF path for tokenizer.ggml.tokens when weights are .4dai
	Parameters     map[string]string
	Template       string
}

// ParseModelfile reads Ollama Modelfile syntax (subset): FROM, PARAMETER, TEMPLATE blocks.
func ParseModelfile(r io.Reader) (Modelfile, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return Modelfile{}, err
	}
	return ParseModelfileString(string(b)), nil
}

// ParseModelfileFile parses a Modelfile from disk.
func ParseModelfileFile(path string) (Modelfile, error) {
	f, err := os.Open(path)
	if err != nil {
		return Modelfile{}, err
	}
	defer f.Close()
	return ParseModelfile(f)
}

func ParseModelfileString(src string) Modelfile {
	var m Modelfile
	m.Parameters = make(map[string]string)
	lines := strings.Split(src, "\n")
	inTemplate := false
	var tpl strings.Builder
	for _, line := range lines {
		if inTemplate {
			if strings.HasPrefix(strings.TrimSpace(line), "\"\"\"") {
				inTemplate = false
				m.Template = strings.TrimSpace(tpl.String())
				tpl.Reset()
				continue
			}
			tpl.WriteString(line)
			tpl.WriteByte('\n')
			continue
		}
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		upper := strings.ToUpper(line)
		if strings.HasPrefix(upper, "FROM ") {
			p := strings.TrimSpace(line[len("FROM "):])
			p = strings.Trim(p, `"'`)
			if p != "" {
				m.FromPaths = append(m.FromPaths, p)
			}
			continue
		}
		if strings.HasPrefix(upper, "TOKENIZER_FROM ") {
			p := strings.TrimSpace(line[len("TOKENIZER_FROM "):])
			p = strings.Trim(p, `"'`)
			if p != "" {
				m.TokenizerFrom = p
			}
			continue
		}
		if strings.HasPrefix(upper, "PARAMETER ") {
			rest := strings.TrimSpace(line[len("PARAMETER "):])
			i := strings.IndexByte(rest, ' ')
			if i > 0 {
				k := strings.TrimSpace(rest[:i])
				v := strings.TrimSpace(rest[i+1:])
				m.Parameters[k] = v
			}
			continue
		}
		if strings.HasPrefix(upper, "TEMPLATE ") && strings.Contains(line, `"""`) {
			inTemplate = true
			continue
		}
	}
	return m
}

// ResolveFromPaths makes FROM paths absolute.
// Relative paths are resolved from processCWD (the shell directory where `4dollama create` ran),
// not the Modelfile's directory and not any server working directory — required for portable shards.
// Absolute paths are cleaned and passed through filepath.Abs.
func ResolveFromPaths(paths []string, processCWD string) ([]string, error) {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var joined string
		if filepath.IsAbs(p) {
			joined = filepath.Clean(p)
		} else {
			joined = filepath.Clean(filepath.Join(processCWD, p))
		}
		abs, err := filepath.Abs(joined)
		if err != nil {
			return nil, fmt.Errorf("modelfile FROM %q: %w", p, err)
		}
		out = append(out, abs)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("modelfile: no FROM paths after resolution")
	}
	return out, nil
}
