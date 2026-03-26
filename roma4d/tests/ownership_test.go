package tests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

func tmpPkg(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	man := filepath.Join(dir, "roma4d.toml")
	if err := os.WriteFile(man, []byte("name = \"tmp\"\nversion = \"0.0.1\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func checkSrc(t *testing.T, dir, name, src string) *compiler.CheckResult {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := compiler.CheckFile(dir, p, nil)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestOwnershipUseAfterMoveLinearVar(t *testing.T) {
	dir := tmpPkg(t)
	src := `
class C:
    soa pos: vec4

def f():
    c = C()
    v = c.pos
    x = v
    y = v
    return x
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "UseAfterMoveError") && strings.Contains(e, "v") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected UseAfterMoveError for second use of linear var, got: %v", res.Errors)
	}
}

func TestOwnershipSoaFieldMoveAndReset(t *testing.T) {
	dir := tmpPkg(t)
	src := `
class C:
    soa pos: vec4

def ok():
    c = C()
    a = c.pos
    c.pos = vec4(x=0, y=0, z=0, w=1)
    b = c.pos
    return a
`
	res := checkSrc(t, dir, "m.roma4d", src)
	if len(res.Errors) > 0 {
		t.Fatalf("expected no errors, got: %v", res.Errors)
	}
}

func TestOwnershipSoaUseAfterMove(t *testing.T) {
	dir := tmpPkg(t)
	src := `
class C:
    soa pos: vec4

def bad():
    c = C()
    a = c.pos
    b = c.pos
    return (a, b)
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "UseAfterMoveError") && strings.Contains(e, "pos") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected UseAfterMoveError for second soa read, got: %v", res.Errors)
	}
}

func TestOwnershipDoubleMutBorrow(t *testing.T) {
	dir := tmpPkg(t)
	src := `
def f():
    v = vec4(x=1, y=0, z=0, w=0)
    mutborrow(v)
    mutborrow(v)
    return v
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "double mutable borrow") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected double mutborrow error, got: %v", res.Errors)
	}
}

func TestOwnershipMutBorrowConflictsImm(t *testing.T) {
	dir := tmpPkg(t)
	src := `
def f():
    v = vec4(x=1, y=0, z=0, w=0)
    borrow(v)
    mutborrow(v)
    return v
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "reborrow conflict") || strings.Contains(e, "immutable borrows") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected imm vs mut borrow error, got: %v", res.Errors)
	}
}

func TestParForNonSendableCapture(t *testing.T) {
	dir := tmpPkg(t)
	src := `
class P:
    soa x: vec4

def f():
    p = P()
    par for i in range(3):
        _ = p
    return 0
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "ParallelismError") && strings.Contains(e, "Sendable") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected ParallelismError for non-Sendable capture, got: %v", res.Errors)
	}
}

func TestParForBorrowExcludesCapture(t *testing.T) {
	dir := tmpPkg(t)
	src := `
def f():
    v = vec4(x=1, y=0, z=0, w=0)
    par for i in range(3):
        _ = borrow(v)
    return v
`
	res := checkSrc(t, dir, "m.roma4d", src)
	if len(res.Errors) > 0 {
		t.Fatalf("borrow(v) should not count as par capture of v, got: %v", res.Errors)
	}
}

func TestParForSendableRotor(t *testing.T) {
	dir := tmpPkg(t)
	src := `
def f():
    pos = [vec4(x=1, y=0, z=0, w=0)]
    r = rotor(angle=1.0, plane="xy")
    par for p in pos:
        _ = p * r
    return 0
`
	res := checkSrc(t, dir, "m.roma4d", src)
	if len(res.Errors) > 0 {
		t.Fatalf("expected rotor capture ok, got: %v", res.Errors)
	}
}

func TestPythonTaintLinearAssign(t *testing.T) {
	dir := tmpPkg(t)
	src := `
class C:
    soa pos: vec4

def f():
    c = C()
    v = vec4(x=1, y=0, z=0, w=0)
    print(v)
    c.pos = v
    return c
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "TaintError") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected TaintError assigning tainted value into soa slot, got: %v", res.Errors)
	}
}

func TestPythonTaintParCapture(t *testing.T) {
	dir := tmpPkg(t)
	src := `
def f():
    r = rotor(angle=1.0, plane="xy")
    print(r)
    par for i in range(2):
        _ = r
    return 0
`
	res := checkSrc(t, dir, "m.roma4d", src)
	var found bool
	for _, e := range res.Errors {
		if strings.Contains(e, "ParallelismError") && strings.Contains(e, "Python interop") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected ParallelismError for tainted par capture, got: %v", res.Errors)
	}
}
