package tests

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
	"github.com/RomanAILabs-Auth/Roma4D/src/parser"
)

func TestParseSpacetimeSyntax(t *testing.T) {
	src := `def f() -> None:
    x: time = t
    y: vec4 = p @ t
    spacetime:
        z = timetravel_borrow(p)
`
	m, err := parser.Parse("spacetime.r4d", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Body) == 0 {
		t.Fatal("empty module")
	}
	fd, ok := m.Body[0].(*parser.FunctionDef)
	if !ok {
		t.Fatalf("expected function, got %T", m.Body[0])
	}
	var hasTime, hasAt, hasST bool
	for _, st := range fd.Body {
		switch s := st.(type) {
		case *parser.AnnAssign:
			if s.Value != nil {
				if _, ok := s.Value.(*parser.TimeCoord); ok {
					hasTime = true
				}
				if bo, ok := s.Value.(*parser.BinOp); ok && bo.Op == parser.AT {
					hasAt = true
				}
			}
		case *parser.SpacetimeStmt:
			hasST = true
		}
	}
	if !hasTime {
		t.Error("expected `t` time coordinate in body")
	}
	if !hasAt {
		t.Error("expected `@` temporal projection p @ t")
	}
	if !hasST {
		t.Error("expected spacetime: region")
	}
}

func TestSpacetimeMIRMarkers(t *testing.T) {
	root := roma4dRoot(t)
	ex := filepath.Join(root, "examples", "hello_4d.r4d")
	res, err := compiler.CheckFile(root, ex, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatal(res.Errors)
	}
	mir, errs := compiler.LowerToMIR(res)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	s := mir.String()
	for _, sub := range []string{
		"temporal_move",
		"spacetime_region",
		"timetravel_borrow",
		"compiletime_temporal_view",
		"ct_r=",
		"gpu_spacetime_par",
	} {
		if !strings.Contains(s, sub) {
			t.Errorf("MIR dump missing %q", sub)
		}
	}
}
