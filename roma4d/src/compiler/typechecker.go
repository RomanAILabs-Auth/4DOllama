package compiler

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/RomanAILabs-Auth/Roma4D/src/parser"
)

// CheckResult holds the outcome of checking one compilation unit.
type CheckResult struct {
	Module    *parser.Module
	RootScope *Scope
	Manifest  *Manifest
	Errors    []string
}

// CheckFile parses and type-checks filename using pkgRoot as the roma4d.toml directory.
// If bench is non-nil, append timings for load_manifest, read_source, parse, typecheck, ownership_pass.
func CheckFile(pkgRoot, filename string, bench *BuildBench) (*CheckResult, error) {
	t := time.Now()
	manPath := filepath.Join(pkgRoot, "roma4d.toml")
	man, err := LoadManifest(manPath)
	if bench != nil {
		bench.Add("load_manifest", time.Since(t))
		t = time.Now()
	}
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	rootAbs, err := filepath.Abs(pkgRoot)
	if err != nil {
		return nil, err
	}
	srcBytes := mustRead(abs)
	if bench != nil {
		bench.Add("read_source", time.Since(t))
		t = time.Now()
	}
	mod, err := parser.Parse(abs, string(srcBytes))
	if bench != nil {
		bench.Add("parse", time.Since(t))
		t = time.Now()
	}
	if err != nil {
		return nil, err
	}
	mod.Qual = qualifyModule(man.Name, rootAbs, abs)

	builtin := NewScope(nil, "builtins")
	seedBuiltins(builtin)

	global := NewScope(builtin, "module "+mod.Qual)
	global.IsModule = true

	ch := &checker{
		rootDir:  rootAbs,
		manifest: man,
		module:   mod,
		scope:    global,
		builtins: builtin,
		loaded:   make(map[string]*ModuleType),
	}
	ch.checkModule(mod)
	if bench != nil {
		bench.Add("typecheck", time.Since(t))
		t = time.Now()
	}
	ch.errs = append(ch.errs, RunOwnershipPass(abs, mod, global)...)
	if bench != nil {
		bench.Add("ownership_pass", time.Since(t))
	}

	return &CheckResult{
		Module:    mod,
		RootScope: global,
		Manifest:  man,
		Errors:    ch.errs,
	}, nil
}

func mustRead(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	return b
}

func qualifyModule(manifestName, rootAbs, fileAbs string) string {
	rel, err := filepath.Rel(rootAbs, fileAbs)
	if err != nil {
		return "__main__"
	}
	rel = strings.TrimSuffix(rel, filepath.Ext(rel))
	rel = filepath.ToSlash(rel)
	parts := strings.Split(rel, "/")
	for i, p := range parts {
		parts[i] = strings.TrimSpace(p)
	}
	q := strings.Join(parts, ".")
	if manifestName != "" {
		return manifestName + "." + q
	}
	return q
}

type checker struct {
	rootDir  string
	manifest *Manifest
	module   *parser.Module
	scope    *Scope
	builtins *Scope
	errs     []string
	loaded   map[string]*ModuleType
}

func (c *checker) errorf(format string, args ...interface{}) {
	c.errs = append(c.errs, fmt.Sprintf(format, args...))
}

// finishDef attaches lexical scope and Sendable (Ownership 2.0) before registration.
func (c *checker) finishDef(sym *Symbol) {
	if sym == nil {
		return
	}
	sym.DefScope = c.scope
	switch sym.Kind {
	case SymVar, SymParam, SymImport:
		sym.Sendable = TypeIsSendable(sym.Type)
	default:
		sym.Sendable = true
	}
}

func (c *checker) checkModule(m *parser.Module) {
	for _, st := range m.Body {
		c.checkStmt(st)
	}
}

func (c *checker) checkStmt(st parser.Stmt) {
	switch s := st.(type) {
	case *parser.FunctionDef:
		c.checkFunctionDef(s, false)
	case *parser.ClassDef:
		c.checkClassDef(s)
	case *parser.ReturnStmt:
		for _, e := range s.Values {
			c.inferExpr(e)
		}
	case *parser.AssignStmt:
		valT := c.inferExpr(s.Value)
		for _, t := range s.Targets {
			c.assignTarget(t, s.Op, valT)
		}
	case *parser.AnnAssign:
		c.checkAnnAssign(s)
	case *parser.ExprStmt:
		c.inferExpr(s.Value)
	case *parser.IfStmt:
		c.inferExpr(s.Test)
		c.pushBlock("if")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
		for _, e := range s.Elifs {
			c.inferExpr(e.Test)
			c.pushBlock("elif")
			for _, x := range e.Body {
				c.checkStmt(x)
			}
			c.popBlock()
		}
		if len(s.Else) > 0 {
			c.pushBlock("else")
			for _, x := range s.Else {
				c.checkStmt(x)
			}
			c.popBlock()
		}
	case *parser.WhileStmt:
		c.inferExpr(s.Test)
		c.pushBlock("while")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
	case *parser.ForStmt:
		c.checkFor(false, s.Target, s.Iter, s.Body, s.Else)
	case *parser.ParForStmt:
		c.checkFor(true, s.Target, s.Iter, s.Body, s.Else)
	case *parser.UnsafeStmt:
		c.pushBlock("unsafe")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
	case *parser.SpacetimeStmt:
		c.pushBlock("spacetime")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
	case *parser.ImportStmt:
		c.checkImport(s)
	case *parser.ImportFrom:
		c.checkImportFrom(s)
	case *parser.PassStmt, *parser.BreakStmt, *parser.ContinueStmt:
	case *parser.TryStmt, *parser.WithStmt, *parser.MatchStmt, *parser.TypeAliasStmt:
		// Pass 3 v0: walk bodies only lightly
		c.walkGenericStmt(st)
	case *parser.GlobalStmt, *parser.NonlocalStmt, *parser.DeleteStmt, *parser.AssertStmt, *parser.RaiseStmt:
	default:
		c.walkGenericStmt(st)
	}
}

func (c *checker) walkGenericStmt(st parser.Stmt) {
	switch s := st.(type) {
	case *parser.TryStmt:
		c.pushBlock("try")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
		for _, h := range s.Handlers {
			c.pushBlock("except")
			for _, x := range h.Body {
				c.checkStmt(x)
			}
			c.popBlock()
		}
		if len(s.Else) > 0 {
			c.pushBlock("else")
			for _, x := range s.Else {
				c.checkStmt(x)
			}
			c.popBlock()
		}
		if len(s.Finally) > 0 {
			c.pushBlock("finally")
			for _, x := range s.Finally {
				c.checkStmt(x)
			}
			c.popBlock()
		}
	case *parser.WithStmt:
		for _, it := range s.Items {
			c.inferExpr(it.Context)
			if it.Target != nil {
				c.inferExpr(it.Target)
			}
		}
		c.pushBlock("with")
		for _, x := range s.Body {
			c.checkStmt(x)
		}
		c.popBlock()
	case *parser.MatchStmt:
		c.inferExpr(s.Subject)
		for _, cs := range s.Cases {
			if cs.Guard != nil {
				c.inferExpr(cs.Guard)
			}
			c.pushBlock("case")
			for _, x := range cs.Body {
				c.checkStmt(x)
			}
			c.popBlock()
		}
	case *parser.TypeAliasStmt:
		ty := c.typeFromAnnotation(s.Value)
		sym := &Symbol{Name: s.Name, Kind: SymTypeAlias, Type: ty, Decl: s}
		c.finishDef(sym)
		if err := c.scope.Define(sym); err != nil {
			c.errorf("%v", err)
		}
	}
}

func (c *checker) pushBlock(label string) {
	c.scope = NewScope(c.scope, label)
}

func (c *checker) popBlock() {
	if c.scope.Parent != nil {
		c.scope = c.scope.Parent
	}
}

func (c *checker) checkImport(st *parser.ImportStmt) {
	for _, al := range st.Names {
		local := al.Name
		if al.As != "" {
			local = al.As
		}
		mt, err := c.resolveModule(al.Name)
		if err != nil {
			c.errorf("ImportError: No module named %q (package root %q; check roma4d.toml `name` and file layout): %v", al.Name, c.rootDir, err)
			continue
		}
		sym := &Symbol{Name: local, Kind: SymImport, Type: mt, ImportPath: al.Name, Decl: st}
		c.finishDef(sym)
		if err := c.scope.Define(sym); err != nil {
			c.errorf("%v", err)
		}
	}
}

func (c *checker) checkImportFrom(st *parser.ImportFrom) {
	if st.Star {
		c.errorf("import * is not supported in Roma4D Pass 3")
		return
	}
	modPath := st.Module
	mt, err := c.resolveModule(modPath)
	if err != nil {
		c.errorf("ImportError: cannot import name from %q: %v", modPath, err)
		return
	}
	for _, al := range st.Names {
		n := al.Name
		local := n
		if al.As != "" {
			local = al.As
		}
		ty, ok := mt.Exports[n]
		if !ok {
			c.errorf("ImportError: cannot import name %q from %q", n, modPath)
			continue
		}
		sym := &Symbol{Name: local, Kind: SymVar, Type: ty, ImportPath: modPath + "." + n, Decl: st}
		c.finishDef(sym)
		if err := c.scope.Define(sym); err != nil {
			c.errorf("%v", err)
		}
	}
}

func (c *checker) resolveModule(dotPath string) (*ModuleType, error) {
	if m, ok := c.loaded[dotPath]; ok {
		return m, nil
	}
	candidates := []string{
		filepath.Join(c.rootDir, strings.ReplaceAll(dotPath, ".", string(filepath.Separator))+".roma4d"),
		filepath.Join(c.rootDir, dotPath+".roma4d"),
	}
	var path string
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			path = p
			break
		}
	}
	if path == "" {
		return nil, fmt.Errorf("module file not found (tried .roma4d under package root)")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	sub, err := parser.Parse(path, string(b))
	if err != nil {
		return nil, err
	}
	sub.Qual = qualifyModule(c.manifest.Name, c.rootDir, path)
	mt := &ModuleType{
		Qual:    sub.Qual,
		Exports: make(map[string]Type),
	}
	// Collect top-level callables and classes without full recursive check (v0).
	for _, st := range sub.Body {
		switch t := st.(type) {
		case *parser.FunctionDef:
			ft := c.signatureFromFunc(t)
			mt.Exports[t.Name] = ft
		case *parser.ClassDef:
			cls := c.classSkeleton(t)
			mt.Exports[t.Name] = cls
		}
	}
	c.loaded[dotPath] = mt
	return mt, nil
}

func (c *checker) signatureFromFunc(fd *parser.FunctionDef) Type {
	params := make([]Type, 0, len(fd.Args.Args))
	for _, a := range fd.Args.Args {
		var pt Type = TypAny
		if a.Annot != nil {
			pt = c.typeFromAnnotation(a.Annot)
		}
		params = append(params, pt)
	}
	ret := TypAny
	if fd.Returns != nil {
		ret = c.typeFromAnnotation(fd.Returns)
	}
	return &Callable{Params: params, Result: ret}
}

func (c *checker) classSkeleton(cd *parser.ClassDef) *Class {
	cls := &Class{Name: cd.Name, Fields: make(map[string]*Field), BaseNames: nil}
	for _, b := range cd.Bases {
		if nm, ok := b.(*parser.Name); ok {
			cls.BaseNames = append(cls.BaseNames, nm.Id)
		}
	}
	return cls
}

func (c *checker) checkClassDef(cd *parser.ClassDef) {
	cls := &Class{Name: cd.Name, Fields: make(map[string]*Field), BaseNames: nil}
	for _, b := range cd.Bases {
		if nm, ok := b.(*parser.Name); ok {
			cls.BaseNames = append(cls.BaseNames, nm.Id)
		}
	}
	classSym := &Symbol{Name: cd.Name, Kind: SymClass, Type: cls, Decl: cd}
	c.finishDef(classSym)
	if err := c.scope.Define(classSym); err != nil {
		c.errorf("%v", err)
	}
	cd.Sym = classSym
	cd.Typ = cls

	classScope := NewScope(c.scope, "class "+cd.Name)
	classScope.IsClass = true
	old := c.scope
	c.scope = classScope

	for _, mem := range cd.Body {
		switch m := mem.(type) {
		case *parser.AnnAssign:
			c.checkClassField(cls, m)
		case *parser.FunctionDef:
			methType := c.signatureFromFunc(m)
			cls.Fields[m.Name] = &Field{Name: m.Name, Type: methType, Layout: LayoutNone}
			fs := &Symbol{Name: m.Name, Kind: SymFunc, Type: methType, Decl: m}
			c.finishDef(fs)
			if err := c.scope.Define(fs); err != nil {
				c.errorf("%v", err)
			}
			m.Sym = fs
			m.Typ = methType
			c.checkFunctionDef(m, true)
		case *parser.ExprStmt, *parser.PassStmt:
		default:
			c.checkStmt(mem)
		}
	}
	c.scope = old
}

func (c *checker) checkClassField(cls *Class, aa *parser.AnnAssign) {
	tn, ok := aa.Target.(*parser.Name)
	if !ok {
		c.errorf("class body annotation target must be a simple name")
		return
	}
	ft := c.typeFromAnnotation(aa.Annotation)
	layout := LayoutNone
	if aa.Layout != nil {
		switch *aa.Layout {
		case parser.KwSoa:
			layout = LayoutSoa
		case parser.KwAos:
			layout = LayoutAos
		}
	}
	f := &Field{Name: tn.Id, Type: ft, Layout: layout, Mutable: true}
	cls.Fields[tn.Id] = f
	sym := &Symbol{Name: tn.Id, Kind: SymField, Type: ft, Decl: aa, LinearHint: layout == LayoutSoa}
	c.finishDef(sym)
	if err := c.scope.Define(sym); err != nil {
		c.errorf("%v", err)
	}
	aa.Sym = sym
	aa.Typ = ft
	if aa.Value != nil {
		vt := c.inferExpr(aa.Value)
		if !c.assignable(ft, vt) {
			c.errorf("TypeError: field %q default value type %s is not compatible with %s", tn.Id, typeStr(vt), typeStr(ft))
		}
	}
}

func (c *checker) checkAnnAssign(aa *parser.AnnAssign) {
	if _, ok := aa.Target.(*parser.Name); !ok {
		c.errorf("annotated assignment target must be a simple name in Pass 3")
		return
	}
	ft := c.typeFromAnnotation(aa.Annotation)
	tn := aa.Target.(*parser.Name)
	var vt Type
	if aa.Value != nil {
		vt = c.inferExpr(aa.Value)
	}
	if isAny(ft) && vt != nil && !isAny(vt) {
		ft = vt
	}
	stored := ft
	if aa.Value != nil && c.assignable(ft, vt) {
		if IsLinearType(vt) {
			stored = vt
		} else if _, ok := vt.(SpacetimeType); ok {
			stored = vt
		}
	}
	discard := tn.Id == DiscardName
	sym := &Symbol{Name: tn.Id, Kind: SymVar, Type: stored, Decl: aa, LinearHint: aa.Layout != nil, Discard: discard}
	c.finishDef(sym)
	if err := c.scope.Define(sym); err != nil {
		c.errorf("%v", err)
	}
	tn.Sym = sym
	tn.Typ = stored
	aa.Sym = sym
	aa.Typ = stored
	if aa.Value != nil {
		if !c.assignable(ft, vt) {
			c.errorf("TypeError: cannot assign value of type %s to %s (%s)", typeStr(vt), tn.Id, typeStr(ft))
		}
	}
}

func (c *checker) checkFunctionDef(fd *parser.FunctionDef, isMethod bool) {
	if !isMethod {
		ft := c.signatureFromFunc(fd)
		sym := &Symbol{Name: fd.Name, Kind: SymFunc, Type: ft, Decl: fd}
		c.finishDef(sym)
		if err := c.scope.Define(sym); err != nil {
			c.errorf("%v", err)
		}
		fd.Sym = sym
		fd.Typ = ft
	}
	fnScope := NewScope(c.scope, "def "+fd.Name)
	fnScope.IsFunction = true
	old := c.scope
	c.scope = fnScope
	for _, a := range fd.Args.Args {
		pt := TypAny
		if a.Annot != nil {
			pt = c.typeFromAnnotation(a.Annot)
		}
		psym := &Symbol{Name: a.Name, Kind: SymParam, Type: pt, Decl: a}
		c.finishDef(psym)
		if err := c.scope.Define(psym); err != nil {
			c.errorf("%v", err)
		}
	}
	for _, st := range fd.Body {
		c.checkStmt(st)
	}
	c.scope = old
}

func (c *checker) checkFor(par bool, tgt parser.Expr, it parser.Expr, body, els []parser.Stmt) {
	c.inferExpr(it)
	loop := NewScope(c.scope, "for")
	if par {
		loop.Name = "par for"
		loop.IsParLoop = true
	}
	c.scope = loop
	if nm, ok := tgt.(*parser.Name); ok {
		elem := c.elementTypeOfIterable(it)
		sym := &Symbol{Name: nm.Id, Kind: SymVar, Type: elem, Decl: nil}
		c.finishDef(sym)
		c.scope.DefineAllowReplace(sym)
		nm.Sym = sym
		nm.Typ = elem
	}
	for _, st := range body {
		c.checkStmt(st)
	}
	c.scope = c.scope.Parent
	if len(els) > 0 {
		c.pushBlock("for else")
		for _, st := range els {
			c.checkStmt(st)
		}
		c.popBlock()
	}
}

func (c *checker) elementTypeOfIterable(it parser.Expr) Type {
	t := c.inferExpr(it)
	switch v := t.(type) {
	case *List:
		if v.Elem != nil {
			return v.Elem
		}
	case *Callable:
		if v.Result != nil {
			if l, ok := v.Result.(*List); ok && l.Elem != nil {
				return l.Elem
			}
		}
	}
	return TypAny
}

// storedTypeAfterAssign keeps Linear{T} when the binding is or becomes linear (Ownership 2.0).
func (c *checker) storedTypeAfterAssign(declared Type, valType Type) Type {
	if IsLinearType(valType) && c.assignable(declared, valType) {
		return valType
	}
	if l, ok := declared.(Linear); ok {
		if c.assignable(declared, valType) {
			return Linear{Inner: l.Inner}
		}
	}
	return valType
}

func (c *checker) assignTarget(t parser.Expr, op parser.TokenKind, valType Type) {
	if op != parser.EQ {
		c.inferExpr(t)
		return
	}
	switch n := t.(type) {
	case *parser.Name:
		sym := c.scope.Lookup(n.Id)
		if sym == nil {
			stored := valType
			discard := n.Id == DiscardName
			sym = &Symbol{Name: n.Id, Kind: SymVar, Type: stored, Decl: n, Discard: discard}
			c.finishDef(sym)
			_ = c.scope.Define(sym)
		} else if sym.Kind == SymVar || sym.Kind == SymParam {
			if !c.assignable(sym.Type, valType) {
				c.errorf("TypeError: cannot assign %s to %q (declared as %s)", typeStr(valType), n.Id, typeStr(sym.Type))
			}
			stored := c.storedTypeAfterAssign(sym.Type, valType)
			sym.Type = stored
			sym.Sendable = TypeIsSendable(stored)
		}
		n.Sym = sym
		if sym != nil {
			n.Typ = sym.Type
		} else {
			n.Typ = valType
		}
	case *parser.Subscript:
		c.inferExpr(t)
	case *parser.Attribute:
		dstT := c.inferExpr(t)
		if op == parser.EQ && !c.assignable(dstT, valType) {
			c.errorf("TypeError: cannot assign %s to attribute (expected %s)", typeStr(valType), typeStr(dstT))
		}
	default:
		c.inferExpr(t)
	}
}

func (c *checker) assignable(dst, src Type) bool {
	if isAny(dst) || isAny(src) {
		return true
	}
	if st, ok := src.(SpacetimeType); ok {
		return c.assignable(dst, st.Inner)
	}
	if st, ok := dst.(SpacetimeType); ok {
		return c.assignable(st.Inner, src)
	}
	if l, ok := dst.(Linear); ok {
		if typesEqual(l.Inner, src) {
			return true
		}
		return typesEqual(l.Inner, StripLinear(src))
	}
	if l, ok := src.(Linear); ok {
		return typesEqual(dst, l.Inner)
	}
	return typesEqual(dst, src)
}

func isAny(t Type) bool {
	p, ok := t.(*Primitive)
	return ok && p.Kind == PrimAny
}

func typesEqual(a, b Type) bool {
	if a == nil || b == nil {
		return true
	}
	return a.TypeString() == b.TypeString()
}

func typeStr(t Type) string {
	if t == nil {
		return "Unknown"
	}
	return t.TypeString()
}

func (c *checker) typeFromAnnotation(e parser.Expr) Type {
	switch n := e.(type) {
	case *parser.Name:
		return c.typeFromName(n.Id)
	case *parser.Constant:
		if n.Kind == parser.KwNone {
			return TypNone
		}
		return TypAny
	case *parser.Subscript:
		if nm, ok := n.Value.(*parser.Name); ok && nm.Id == "list" {
			el := c.typeFromAnnotation(n.Slice)
			return &List{Elem: el}
		}
		_ = c.typeFromAnnotation(n.Value)
		return TypAny
	default:
		return TypAny
	}
}

func (c *checker) typeFromName(id string) Type {
	switch strings.ToLower(id) {
	case "int":
		return TypInt
	case "float":
		return TypFloat
	case "str":
		return TypStr
	case "bool":
		return TypBool
	case "any":
		return TypAny
	case "none":
		return TypNone
	case "vec4":
		return TypVec4
	case "rotor":
		return TypRotor
	case "multivector":
		return TypMultivector
	case "rawptr":
		return TypRawPtr
	case "time":
		return TypTime
	}
	if sym := c.scope.Lookup(id); sym != nil && sym.Kind == SymTypeAlias {
		return sym.Type
	}
	if sym := c.scope.Lookup(id); sym != nil && sym.Kind == SymClass {
		if cl, ok := sym.Type.(*Class); ok {
			return cl
		}
	}
	return TypAny
}

func (c *checker) inferExpr(e parser.Expr) Type {
	switch n := e.(type) {
	case *parser.Name:
		sym := c.scope.Lookup(n.Id)
		if sym == nil {
			c.errorf("NameError: name %q is not defined", n.Id)
			n.Typ = TypAny
			return TypAny
		}
		n.Sym = sym
		n.Typ = sym.Type
		return sym.Type
	case *parser.Constant:
		var t Type
		switch n.Kind {
		case parser.INT:
			t = TypInt
		case parser.FLOAT:
			t = TypFloat
		case parser.IMAG:
			t = &Primitive{Kind: PrimComplex}
		case parser.STRING:
			t = TypStr
		case parser.KwTrue, parser.KwFalse:
			t = TypBool
		case parser.KwNone:
			t = TypNone
		default:
			t = TypAny
		}
		n.Typ = t
		return t
	case *parser.TimeCoord:
		n.Typ = TypTime
		return TypTime
	case *parser.BinOp:
		lt := c.inferExpr(n.Left)
		rt := c.inferExpr(n.Right)
		out := c.typeBinOp(n, n.Op, lt, rt)
		n.Typ = out
		return out
	case *parser.UnaryOp:
		t := c.inferExpr(n.Expr)
		return t
	case *parser.Compare:
		c.inferExpr(n.Left)
		for _, x := range n.Comparators {
			c.inferExpr(x)
		}
		return TypBool
	case *parser.BoolOp:
		for _, v := range n.Values {
			c.inferExpr(v)
		}
		return TypBool
	case *parser.IfExp:
		c.inferExpr(n.Test)
		a := c.inferExpr(n.Body)
		b := c.inferExpr(n.Orelse)
		if typesEqual(a, b) {
			return a
		}
		return TypAny
	case *parser.Call:
		return c.inferCall(n)
	case *parser.Attribute:
		vt := c.inferExpr(n.Value)
		if cls, ok := vt.(*Class); ok {
			if f, ok := cls.Fields[n.Attr]; ok {
				ft := f.Type
				if f.Layout == LayoutSoa {
					ft = Linear{Inner: ft}
				}
				n.Typ = ft
				return ft
			}
		}
		c.errorf("AttributeError: %s has no attribute %q", typeStr(vt), n.Attr)
		n.Typ = TypAny
		return TypAny
	case *parser.Subscript:
		vt := c.inferExpr(n.Value)
		c.inferExpr(n.Slice)
		if lst, ok := vt.(*List); ok && lst.Elem != nil {
			n.Typ = lst.Elem
			return lst.Elem
		}
		n.Typ = TypAny
		return TypAny
	case *parser.ListComp:
		et := c.inferExpr(n.Elt)
		return &List{Elem: et}
	case *parser.ListExpr:
		var merge Type = TypAny
		for _, el := range n.Elts {
			t := c.inferExpr(el)
			if merge == TypAny {
				merge = t
			} else if !typesEqual(merge, t) {
				merge = TypAny
			}
		}
		return &List{Elem: merge}
	case *parser.TupleExpr:
		ts := make([]Type, len(n.Elts))
		for i, el := range n.Elts {
			ts[i] = c.inferExpr(el)
		}
		return &Tuple{Elts: ts}
	case *parser.DictExpr, *parser.SetExpr, *parser.Lambda, *parser.AwaitExpr:
		return TypAny
	default:
		return TypAny
	}
}

func (c *checker) inferCall(n *parser.Call) Type {
	// Callee symbol for builtins / functions
	switch fn := n.Func.(type) {
	case *parser.Name:
		sym := c.scope.Lookup(fn.Id)
		if sym != nil {
			n.Sym = sym
		}
		switch fn.Id {
		case "range":
			for _, a := range n.Args {
				c.inferExpr(a)
			}
			n.Typ = &List{Elem: TypInt}
			return &List{Elem: TypInt}
		case "len":
			for _, a := range n.Args {
				c.inferExpr(a)
			}
			n.Typ = TypInt
			return TypInt
		case "print":
			for _, a := range n.Args {
				c.inferExpr(a)
			}
			for _, k := range n.Keywords {
				c.inferExpr(k.Val)
			}
			n.Typ = TypNone
			return TypNone
		case "borrow":
			if len(n.Args) != 1 {
				c.errorf("borrow() expects exactly one argument")
				n.Typ = TypAny
				return TypAny
			}
			inner := c.inferExpr(n.Args[0])
			br := &BorrowedRef{Inner: StripLinear(inner)}
			n.Typ = br
			return br
		case "mutborrow":
			if len(n.Args) != 1 {
				c.errorf("mutborrow() expects exactly one argument")
				n.Typ = TypAny
				return TypAny
			}
			inner := c.inferExpr(n.Args[0])
			mb := &MutBorrowRef{Inner: StripLinear(inner)}
			n.Typ = mb
			return mb
		case "timetravel_borrow":
			if len(n.Args) != 1 {
				c.errorf("timetravel_borrow() expects exactly one argument")
				n.Typ = TypAny
				return TypAny
			}
			inner := c.inferExpr(n.Args[0])
			br := &BorrowedRef{Inner: StripLinear(inner)}
			n.Typ = br
			return br
		case "vec4":
			for _, k := range n.Keywords {
				c.inferExpr(k.Val)
			}
			n.Typ = TypVec4
			return TypVec4
		case "rotor":
			for _, k := range n.Keywords {
				c.inferExpr(k.Val)
			}
			n.Typ = TypRotor
			return TypRotor
		case "multivector":
			for _, a := range n.Args {
				c.inferExpr(a)
			}
			for _, k := range n.Keywords {
				c.inferExpr(k.Val)
			}
			n.Typ = TypMultivector
			return TypMultivector
		case "mir_alloc":
			if len(n.Args) != 1 {
				c.errorf("mir_alloc() expects exactly one argument (byte size)")
				n.Typ = TypRawPtr
				return TypRawPtr
			}
			c.inferExpr(n.Args[0])
			n.Typ = TypRawPtr
			return TypRawPtr
		case "mir_ptr_load":
			if len(n.Args) != 1 {
				c.errorf("mir_ptr_load() expects exactly one argument (address)")
				n.Typ = TypInt
				return TypInt
			}
			c.inferExpr(n.Args[0])
			n.Typ = TypInt
			return TypInt
		case "ollama_demo":
			if len(n.Args) != 0 {
				c.errorf("ollama_demo() expects no arguments (fixed JSON body in rt/roma4d_rt.c)")
				n.Typ = TypInt
				return TypInt
			}
			n.Typ = TypInt
			return TypInt
		case "quantum_server_demo":
			if len(n.Args) != 0 {
				c.errorf("quantum_server_demo() expects no arguments (see rt/roma4d_rt.c; use env QUANTUM_QUERY)")
				n.Typ = TypInt
				return TypInt
			}
			n.Typ = TypInt
			return TypInt
		case "mir_ptr_store":
			if len(n.Args) != 2 {
				c.errorf("mir_ptr_store() expects (address, int value)")
				n.Typ = TypNone
				return TypNone
			}
			c.inferExpr(n.Args[0])
			c.inferExpr(n.Args[1])
			n.Typ = TypNone
			return TypNone
		case "int", "float", "str", "bool":
			if len(n.Args) > 0 {
				c.inferExpr(n.Args[0])
			}
			var res Type = TypAny
			switch fn.Id {
			case "int":
				res = TypInt
			case "float":
				res = TypFloat
			case "str":
				res = TypStr
			case "bool":
				res = TypBool
			}
			n.Typ = res
			return res
		}
		if sym != nil {
			if cl, ok := sym.Type.(*Callable); ok && cl.Result != nil {
				for _, a := range n.Args {
					c.inferExpr(a)
				}
				for _, k := range n.Keywords {
					c.inferExpr(k.Val)
				}
				n.Typ = cl.Result
				return cl.Result
			}
			if sym.Kind == SymClass {
				if cls, ok := sym.Type.(*Class); ok {
					for _, a := range n.Args {
						c.inferExpr(a)
					}
					for _, k := range n.Keywords {
						c.inferExpr(k.Val)
					}
					n.Typ = cls
					return cls
				}
			}
		}
		c.inferExpr(n.Func)
		for _, a := range n.Args {
			c.inferExpr(a)
		}
		for _, k := range n.Keywords {
			c.inferExpr(k.Val)
		}
		n.Typ = TypAny
		return TypAny
	default:
		ft := c.inferExpr(n.Func)
		for _, a := range n.Args {
			c.inferExpr(a)
		}
		for _, k := range n.Keywords {
			c.inferExpr(k.Val)
		}
		if cl, ok := ft.(*Callable); ok && cl.Result != nil {
			n.Typ = cl.Result
			return cl.Result
		}
		n.Typ = TypAny
		return TypAny
	}
}

func (c *checker) typeBinOp(n *parser.BinOp, op parser.TokenKind, L, R Type) Type {
	switch op {
	case parser.AT:
		if _, ok := R.(TimeDim); ok {
			tag := "t"
			return SpacetimeType{Inner: StripLinear(L), TimeTag: tag}
		}
		c.errorf("TypeError: `@` (temporal projection) requires right-hand `t` (time), got %s", typeStr(R))
		return TypAny
	case parser.STAR:
		if isNumeric(L) && isNumeric(R) {
			return PromoteNumeric(L, R)
		}
		if isGA(L) || isGA(R) {
			if isAny(L) || isAny(R) {
				return TypAny
			}
			return c.geometricProduct(L, R)
		}
		c.errorf("TypeError: unsupported operand type(s) for *: %s and %s", typeStr(L), typeStr(R))
		return TypAny
	case parser.CARET:
		if isIntLike(L) && isIntLike(R) {
			return TypInt
		}
		if isGA(L) || isGA(R) {
			return TypMultivector
		}
		c.errorf("TypeError: unsupported operand type(s) for ^: %s and %s", typeStr(L), typeStr(R))
		return TypAny
	case parser.PIPE:
		if isIntLike(L) && isIntLike(R) {
			return TypInt
		}
		if isGA(L) || isGA(R) {
			return TypFloat
		}
		c.errorf("TypeError: unsupported operand type(s) for |: %s and %s", typeStr(L), typeStr(R))
		return TypAny
	case parser.PLUS, parser.MINUS:
		if isNumeric(L) && isNumeric(R) {
			return PromoteNumeric(L, R)
		}
		if (isVec4(L) && isVec4(R)) && op == parser.PLUS {
			return TypVec4
		}
		return TypAny
	case parser.SLASH:
		if isNumeric(L) && isNumeric(R) {
			return TypFloat
		}
		return TypAny
	case parser.PERCENT, parser.DBLSLASH:
		if isIntLike(L) && isIntLike(R) {
			return TypInt
		}
		return TypAny
	case parser.AMP, parser.LSHIFT, parser.RSHIFT:
		if isIntLike(L) && isIntLike(R) {
			return TypInt
		}
		return TypAny
	default:
		return TypAny
	}
}

func isVec4(t Type) bool {
	_, ok := StripSpacetime(StripLinear(t)).(Vec4)
	return ok
}

func isRotor(t Type) bool {
	_, ok := StripSpacetime(StripLinear(t)).(Rotor)
	return ok
}

func (c *checker) geometricProduct(L, R Type) Type {
	lv, rv := isVec4(L), isVec4(R)
	lr, rr := isRotor(L), isRotor(R)
	lm, rm := isMultivector(L), isMultivector(R)

	if lv && rv {
		return TypMultivector
	}
	if lv && rr {
		return TypVec4
	}
	if lr && rv {
		return TypMultivector
	}
	if lm || rm {
		return TypMultivector
	}
	if (lv || lr) && (rv || rr) {
		return TypMultivector
	}
	c.errorf("TypeError: geometric product not defined for %s and %s", typeStr(L), typeStr(R))
	return TypMultivector
}

func isMultivector(t Type) bool {
	_, ok := StripSpacetime(StripLinear(t)).(Multivector)
	return ok
}

func isIntLike(t Type) bool {
	p, ok := t.(*Primitive)
	return ok && p.Kind == PrimInt
}

func seedBuiltins(s *Scope) {
	add := func(name string, sym *Symbol) {
		sym.DefScope = nil
		if cl, ok := sym.Type.(*Callable); ok && cl.Result != nil {
			sym.Sendable = TypeIsSendable(cl.Result)
		} else if sym.Type != nil {
			sym.Sendable = TypeIsSendable(sym.Type)
		} else {
			sym.Sendable = true
		}
		s.DefineAllowReplace(sym)
	}
	add("print", &Symbol{Name: "print", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypNone}})
	add("range", &Symbol{Name: "range", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: &List{Elem: TypInt}}})
	add("len", &Symbol{Name: "len", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypAny}, Result: TypInt}})
	add("int", &Symbol{Name: "int", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypInt, IsCtor: true}})
	add("float", &Symbol{Name: "float", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypFloat, IsCtor: true}})
	add("str", &Symbol{Name: "str", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypStr, IsCtor: true}})
	add("bool", &Symbol{Name: "bool", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypBool, IsCtor: true}})
	add("abs", &Symbol{Name: "abs", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypAny}, Result: TypAny}})
	add("borrow", &Symbol{Name: "borrow", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypAny}, Result: TypAny}})
	add("mutborrow", &Symbol{Name: "mutborrow", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypAny}, Result: TypAny}})
	add("timetravel_borrow", &Symbol{Name: "timetravel_borrow", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypAny}, Result: TypAny}})
	add("vec4", &Symbol{Name: "vec4", Kind: SymBuiltin, Type: &Callable{KwVariadic: true, Result: TypVec4, IsCtor: true}})
	add("rotor", &Symbol{Name: "rotor", Kind: SymBuiltin, Type: &Callable{KwVariadic: true, Result: TypRotor, IsCtor: true}})
	add("multivector", &Symbol{Name: "multivector", Kind: SymBuiltin, Type: &Callable{Variadic: true, Result: TypMultivector, IsCtor: true}})
	add("mir_alloc", &Symbol{Name: "mir_alloc", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypInt}, Result: TypRawPtr}})
	add("mir_ptr_load", &Symbol{Name: "mir_ptr_load", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypRawPtr}, Result: TypInt}})
	add("mir_ptr_store", &Symbol{Name: "mir_ptr_store", Kind: SymBuiltin, Type: &Callable{Params: []Type{TypRawPtr, TypInt}, Result: TypNone}})
	add("ollama_demo", &Symbol{Name: "ollama_demo", Kind: SymBuiltin, Type: &Callable{Params: nil, Result: TypInt}})
	add("quantum_server_demo", &Symbol{Name: "quantum_server_demo", Kind: SymBuiltin, Type: &Callable{Params: nil, Result: TypInt}})
	add("True", &Symbol{Name: "True", Kind: SymBuiltin, Type: TypBool})
	add("False", &Symbol{Name: "False", Kind: SymBuiltin, Type: TypBool})
	add("None", &Symbol{Name: "None", Kind: SymBuiltin, Type: TypNone})
}
