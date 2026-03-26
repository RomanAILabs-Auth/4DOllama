package parser

import (
	"path/filepath"
	"testing"
)

func TestParseHelloExample(t *testing.T) {
	p := filepath.Join("..", "..", "examples", "hello_4d.r4d")
	m, err := ParseFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Body) == 0 {
		t.Fatal("empty module")
	}
}

func TestParseParForAndVec4(t *testing.T) {
	src := `
class P:
    soa pos: vec4
    aos vel: vec4

def main():
    v = vec4(x=1, y=0, z=0, w=0)
    r = rotor(angle=1.0, plane="xy")
    m = v ^ v
    n = v | v
    par for p in pos:
        p = p * r
`
	m, err := Parse("t.roma4d", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Body) < 2 {
		t.Fatalf("expected at least 2 top-level stmts, got %d", len(m.Body))
	}
}

func TestParseStripsUTF8BOM(t *testing.T) {
	src := "\ufeffdef main():\n    pass\n"
	m, err := Parse("bom.roma4d", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Body) == 0 {
		t.Fatal("expected module body")
	}
	fd, ok := m.Body[0].(*FunctionDef)
	if !ok || fd.Name != "main" {
		t.Fatalf("expected def main, got %#v", m.Body[0])
	}
}

func TestParseStripsDoubleUTF8BOM(t *testing.T) {
	src := "\ufeff\ufeffdef main():\n    pass\n"
	m, err := Parse("bom2.roma4d", src)
	if err != nil {
		t.Fatal(err)
	}
	fd, ok := m.Body[0].(*FunctionDef)
	if !ok || fd.Name != "main" {
		t.Fatalf("expected def main after double BOM, got %#v", m.Body[0])
	}
}

func TestLexerIndent(t *testing.T) {
	src := "if 1:\n    pass\n"
	lx := NewLexer("x", src)
	var kinds []TokenKind
	for {
		tok := lx.Next()
		kinds = append(kinds, tok.Kind)
		if tok.Kind == EOF {
			break
		}
	}
	// if 1 : NEWLINE INDENT pass NEWLINE DEDENT EOF
	foundIndent := false
	foundDedent := false
	for _, k := range kinds {
		if k == INDENT {
			foundIndent = true
		}
		if k == DEDENT {
			foundDedent = true
		}
	}
	if !foundIndent || !foundDedent {
		t.Fatalf("INDENT/DEDENT missing: %#v", kinds)
	}
}
