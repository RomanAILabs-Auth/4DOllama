package tui

import (
	"io"
	"strings"
	"unicode/utf8"
)

// StreamingPlainWriter applies sanitizeChatDisplay incrementally so terminals show plain text
// as tokens arrive (fences unwrap when complete; incomplete fences show inner content).
type StreamingPlainWriter struct {
	Out       io.Writer
	buf       strings.Builder
	displayed int
}

func (w *StreamingPlainWriter) Reset() {
	w.buf.Reset()
	w.displayed = 0
}

func (w *StreamingPlainWriter) WriteChunk(s string) {
	if s == "" || w.Out == nil {
		return
	}
	w.buf.WriteString(s)
	disp := sanitizeChatDisplay(w.buf.String())
	if len(disp) > w.displayed {
		_, _ = io.WriteString(w.Out, disp[w.displayed:])
		w.displayed = len(disp)
	}
}

// sanitizeChatDisplay turns model output into terminal-safe plain text: unwrap fenced code
// blocks (no ``` lines), soften markdown tables, strip heading hashes, normalize newlines.
// Streaming-safe: if the closing ``` is missing, content after the opening fence line is shown as plain.
func sanitizeChatDisplay(s string) string {
	if s == "" {
		return ""
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "")
	var out strings.Builder
	out.Grow(len(s))
	i := 0
	for i < len(s) {
		if i+2 < len(s) && s[i] == '`' && s[i+1] == '`' && s[i+2] == '`' {
			j := i + 3
			for j < len(s) && s[j] != '\n' {
				j++
			}
			if j < len(s) {
				j++
			}
			rest := s[j:]
			k := strings.Index(rest, "```")
			if k >= 0 {
				inner := rest[:k]
				ensureNLBefore(&out)
				out.WriteString(inner)
				if len(inner) > 0 && !strings.HasSuffix(inner, "\n") {
					out.WriteByte('\n')
				}
				i = j + k + 3
				continue
			}
			ensureNLBefore(&out)
			out.WriteString(rest)
			break
		}
		_, rw := utf8.DecodeRuneInString(s[i:])
		if rw == 0 {
			break
		}
		out.WriteString(s[i : i+rw])
		i += rw
	}
	t := out.String()
	t = softenPipeTableLines(t)
	t = stripMarkdownHeadings(t)
	return t
}

func ensureNLBefore(b *strings.Builder) {
	if b.Len() == 0 {
		return
	}
	if !strings.HasSuffix(b.String(), "\n") {
		b.WriteByte('\n')
	}
}

func stripMarkdownHeadings(s string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		t := strings.TrimSpace(line)
		if t == "" || t[0] != '#' {
			continue
		}
		n := 0
		for n < len(t) && t[n] == '#' {
			n++
		}
		if n >= len(t) {
			continue
		}
		if t[n] != ' ' && t[n] != '\t' {
			continue
		}
		lines[i] = strings.TrimSpace(t[n+1:])
	}
	return strings.Join(lines, "\n")
}

// softenPipeTableLines replaces markdown-style table rows (mostly pipes) with a single-line form.
func softenPipeTableLines(s string) string {
	lines := strings.Split(s, "\n")
	for li, line := range lines {
		if !isPipeTableLine(line) {
			continue
		}
		cells := strings.Split(line, "|")
		var parts []string
		for _, c := range cells {
			c = strings.TrimSpace(c)
			if c == "" || strings.Trim(c, "-: ") == "" {
				continue
			}
			parts = append(parts, c)
		}
		if len(parts) > 0 {
			lines[li] = strings.Join(parts, " · ")
		}
	}
	return strings.Join(lines, "\n")
}

func isPipeTableLine(line string) bool {
	t := strings.TrimSpace(line)
	if t == "" || !strings.Contains(t, "|") {
		return false
	}
	pipes := strings.Count(t, "|")
	if pipes < 2 {
		return false
	}
	if strings.Contains(t, "---") {
		return true
	}
	letters := 0
	for _, r := range t {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			letters++
		}
	}
	runes := utf8.RuneCountInString(t)
	if runes == 0 {
		return false
	}
	return letters > 0 && float64(pipes)/float64(runes) >= 0.15
}
