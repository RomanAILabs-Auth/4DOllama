package compiler

// Ownership 2.0 is spacetime-aware: borrows are scoped to lexical blocks and to
// distinct compile-time temporal regions (`spacetime:`). All ordering checks run
// during compilation; the emitted binary has no temporal runtime, no epoch, and
// no borrow instrumentation — only ordinary loads/stores and the same spatial
// rules as a classic systems language.

import (
	"fmt"

	"github.com/RomanAILabs-Auth/Roma4D/src/parser"
)

// RunOwnershipPass performs Ownership 2.0 checking after type checking.
// filename is used in diagnostics; mod and global come from CheckResult.
func RunOwnershipPass(filename string, mod *parser.Module, global *Scope) []string {
	p := &opass{
		file:        filename,
		states:      make(map[*Symbol]*varState),
		linearSlots: make(map[linearSlotKey]struct{}),
	}
	p.pushBorrowFrame()
	for _, st := range mod.Body {
		p.ownStmt(st)
		p.endStmt()
	}
	p.popBorrowFrame()
	return p.errs
}

type linearSlotKey struct {
	base  *Symbol
	field string
}

type opass struct {
	file   string
	errs   []string
	states map[*Symbol]*varState

	// linearSlots marks SoA / linear attribute slots (base var + field) consumed by move.
	linearSlots map[linearSlotKey]struct{}

	// borrowStack: immutable/mutable borrows end at block exit (Systems Edition).
	borrowStack []borrowFrame

	// spacetimeEpoch is a monotonic compile-time region counter for `spacetime:` bodies.
	// It exists only to label diagnostics (e.g. borrow conflicts); it is not runtime state.
	spacetimeEpoch int64

	loopVarSym *Symbol
}

type borrowFrame struct {
	imm    []*Symbol
	mut    []*Symbol
	tEpoch int64 // 0 = outer timeline; >0 = compile-time id for nested `spacetime:` (static only)
}

type varState struct {
	moved      bool
	immBorrows int
	mutActive  bool
}

func (p *opass) errf(line, col int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	if line > 0 {
		p.errs = append(p.errs, fmt.Sprintf("%s:%d:%d: %s", p.file, line, col, msg))
	} else {
		p.errs = append(p.errs, fmt.Sprintf("%s: %s", p.file, msg))
	}
}

func (p *opass) get(sym *Symbol) *varState {
	if sym == nil {
		return nil
	}
	st, ok := p.states[sym]
	if !ok {
		st = &varState{}
		p.states[sym] = st
	}
	return st
}

func (p *opass) endStmt() {
	// Borrows are block-scoped; no per-statement release.
}

func (p *opass) pushBorrowFrame() {
	var te int64
	if n := len(p.borrowStack); n > 0 {
		te = p.borrowStack[n-1].tEpoch
	}
	p.borrowStack = append(p.borrowStack, borrowFrame{tEpoch: te})
}

func (p *opass) pushSpacetimeBorrowFrame() {
	p.spacetimeEpoch++
	te := p.spacetimeEpoch
	p.borrowStack = append(p.borrowStack, borrowFrame{tEpoch: te})
}

func (p *opass) popBorrowFrame() {
	n := len(p.borrowStack)
	if n == 0 {
		return
	}
	fr := p.borrowStack[n-1]
	p.borrowStack = p.borrowStack[:n-1]
	for _, s := range fr.imm {
		st := p.get(s)
		if st != nil && st.immBorrows > 0 {
			st.immBorrows--
		}
	}
	for _, s := range fr.mut {
		st := p.get(s)
		if st != nil {
			st.mutActive = false
		}
	}
}

func (p *opass) curBorrowFrame() *borrowFrame {
	return &p.borrowStack[len(p.borrowStack)-1]
}

func (p *opass) ownStmt(st parser.Stmt) {
	switch s := st.(type) {
	case *parser.FunctionDef:
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.ClassDef:
		p.pushBorrowFrame()
		for _, m := range s.Body {
			p.ownStmt(m)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.AssignStmt:
		for _, t := range s.Targets {
			p.resetLinearSlotTarget(t)
		}
		p.ownExpr(s.Value, consumeMode)
		for _, t := range s.Targets {
			if IsLinearType(ifaceTypeFromExpr(t)) && p.exprHasTainted(s.Value) {
				p.errf(0, 0, "TaintError: cannot assign a Python-tainted value (e.g. after print) into a linear / `soa` slot")
			}
			if nm, ok := t.(*parser.Name); ok {
				if sym, ok := nm.Sym.(*Symbol); ok && sym != nil {
					st := p.get(sym)
					st.moved = false
					st.mutActive = false
					st.immBorrows = 0
				}
			}
		}
	case *parser.AnnAssign:
		p.resetLinearSlotTarget(s.Target)
		if s.Value != nil {
			p.ownExpr(s.Value, consumeMode)
		}
		if nm, ok := s.Target.(*parser.Name); ok {
			if sym, ok := nm.Sym.(*Symbol); ok && sym != nil {
				st := p.get(sym)
				st.moved = false
				st.mutActive = false
				st.immBorrows = 0
			}
		}
	case *parser.ExprStmt:
		if call, ok := s.Value.(*parser.Call); ok {
			if fn, ok := call.Func.(*parser.Name); ok && fn.Id == "print" {
				for _, a := range call.Args {
					p.taintPythonNames(a)
				}
				for _, k := range call.Keywords {
					p.taintPythonNames(k.Val)
				}
			}
		}
		p.ownExpr(s.Value, discardMode)
	case *parser.ReturnStmt:
		for _, e := range s.Values {
			p.ownExpr(e, consumeMode)
		}
	case *parser.IfStmt:
		p.ownExpr(s.Test, discardMode)
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
		for _, e := range s.Elifs {
			p.ownExpr(e.Test, discardMode)
			p.pushBorrowFrame()
			for _, x := range e.Body {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
		if len(s.Else) > 0 {
			p.pushBorrowFrame()
			for _, x := range s.Else {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
	case *parser.WhileStmt:
		p.ownExpr(s.Test, discardMode)
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.ForStmt:
		p.ownLoop(s.Target, s.Iter, s.Body, s.Else, false)
	case *parser.ParForStmt:
		p.ownLoop(s.Target, s.Iter, s.Body, s.Else, true)
	case *parser.ImportStmt, *parser.ImportFrom, *parser.PassStmt, *parser.BreakStmt, *parser.ContinueStmt:
	case *parser.TryStmt:
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
		for _, h := range s.Handlers {
			p.pushBorrowFrame()
			for _, x := range h.Body {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
		if len(s.Else) > 0 {
			p.pushBorrowFrame()
			for _, x := range s.Else {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
		if len(s.Finally) > 0 {
			p.pushBorrowFrame()
			for _, x := range s.Finally {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
	case *parser.WithStmt:
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.MatchStmt:
		for _, c := range s.Cases {
			p.pushBorrowFrame()
			for _, x := range c.Body {
				p.ownStmt(x)
				p.endStmt()
			}
			p.popBorrowFrame()
		}
	case *parser.UnsafeStmt:
		p.pushBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.SpacetimeStmt:
		p.pushSpacetimeBorrowFrame()
		for _, x := range s.Body {
			p.ownStmt(x)
			p.endStmt()
		}
		p.popBorrowFrame()
	case *parser.TypeAliasStmt, *parser.GlobalStmt, *parser.NonlocalStmt, *parser.DeleteStmt, *parser.AssertStmt, *parser.RaiseStmt:
	default:
	}
}

func (p *opass) ownLoop(tgt parser.Expr, it parser.Expr, body, els []parser.Stmt, isPar bool) {
	p.ownExpr(it, discardMode)
	var saved *Symbol
	if nm, ok := tgt.(*parser.Name); ok {
		if sym, ok := nm.Sym.(*Symbol); ok {
			saved = p.loopVarSym
			p.loopVarSym = sym
		}
	}
	if isPar {
		p.checkParCaptures(body)
	}
	p.pushBorrowFrame()
	for _, st := range body {
		p.ownStmt(st)
		p.endStmt()
	}
	p.popBorrowFrame()
	p.loopVarSym = saved
	if len(els) > 0 {
		p.pushBorrowFrame()
		for _, st := range els {
			p.ownStmt(st)
			p.endStmt()
		}
		p.popBorrowFrame()
	}
}

func (p *opass) checkParCaptures(body []parser.Stmt) {
	names := collectNamesForParCaptureStmts(body)
	seen := make(map[*Symbol]struct{})
	for _, n := range names {
		sym, ok := n.Sym.(*Symbol)
		if !ok || sym == nil {
			continue
		}
		if sym == p.loopVarSym {
			continue
		}
		if sym.Kind == SymBuiltin || sym.Kind == SymClass {
			continue
		}
		if _, dup := seen[sym]; dup {
			continue
		}
		seen[sym] = struct{}{}

		et := exprTypeFromAST(n)
		if TypeIsSendable(et) && !sym.TaintedPython {
			continue
		}
		if sym.TaintedPython {
			p.errf(n.Line, n.Col, "ParallelismError: cannot capture %q in `par for`: value was passed to Python interop (e.g. print) and is not safe to share across parallel iterations", n.Id)
			continue
		}
		p.errf(n.Line, n.Col, "ParallelismError: cannot capture %q in `par for`: not Sendable (wrap with borrow(...) for a shared immutable view, or use only Sendable types)", n.Id)
	}
}

func exprTypeFromAST(n *parser.Name) Type {
	if n == nil {
		return TypAny
	}
	if t, ok := n.Typ.(Type); ok && t != nil {
		return t
	}
	return TypAny
}

// collectNamesForParCaptureStmts collects names that are actually captured by `par for`,
// excluding names that only appear inside borrow(...) (shared immutable view).
func collectNamesForParCaptureStmts(stmts []parser.Stmt) []*parser.Name {
	var out []*parser.Name
	for _, st := range stmts {
		out = append(out, collectNamesForParCaptureStmt(st)...)
	}
	return out
}

func collectNamesForParCaptureStmt(st parser.Stmt) []*parser.Name {
	var out []*parser.Name
	switch s := st.(type) {
	case *parser.AssignStmt:
		for _, t := range s.Targets {
			out = append(out, collectNamesForParCaptureExpr(t)...)
		}
		out = append(out, collectNamesForParCaptureExpr(s.Value)...)
	case *parser.AnnAssign:
		out = append(out, collectNamesForParCaptureExpr(s.Target)...)
		out = append(out, collectNamesForParCaptureExpr(s.Annotation)...)
		if s.Value != nil {
			out = append(out, collectNamesForParCaptureExpr(s.Value)...)
		}
	case *parser.ExprStmt:
		out = append(out, collectNamesForParCaptureExpr(s.Value)...)
	case *parser.ReturnStmt:
		for _, e := range s.Values {
			out = append(out, collectNamesForParCaptureExpr(e)...)
		}
	case *parser.IfStmt:
		out = append(out, collectNamesForParCaptureExpr(s.Test)...)
		for _, x := range s.Body {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
		for _, e := range s.Elifs {
			out = append(out, collectNamesForParCaptureExpr(e.Test)...)
			for _, x := range e.Body {
				out = append(out, collectNamesForParCaptureStmt(x)...)
			}
		}
		for _, x := range s.Else {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	case *parser.WhileStmt:
		out = append(out, collectNamesForParCaptureExpr(s.Test)...)
		for _, x := range s.Body {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	case *parser.ForStmt, *parser.ParForStmt:
		var b []parser.Stmt
		var it, tg parser.Expr
		switch v := s.(type) {
		case *parser.ForStmt:
			b, it, tg = v.Body, v.Iter, v.Target
		case *parser.ParForStmt:
			b, it, tg = v.Body, v.Iter, v.Target
		}
		out = append(out, collectNamesForParCaptureExpr(it)...)
		out = append(out, collectNamesForParCaptureExpr(tg)...)
		for _, x := range b {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	case *parser.FunctionDef:
		for _, x := range s.Body {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	case *parser.ClassDef:
		for _, x := range s.Body {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	case *parser.UnsafeStmt:
		for _, x := range s.Body {
			out = append(out, collectNamesForParCaptureStmt(x)...)
		}
	}
	return out
}

func collectNamesForParCaptureExpr(e parser.Expr) []*parser.Name {
	if e == nil {
		return nil
	}
	switch x := e.(type) {
	case *parser.Call:
		if fn, ok := x.Func.(*parser.Name); ok && (fn.Id == "borrow" || fn.Id == "timetravel_borrow") {
			return nil
		}
		var out []*parser.Name
		out = append(out, collectNamesForParCaptureExpr(x.Func)...)
		for _, a := range x.Args {
			out = append(out, collectNamesForParCaptureExpr(a)...)
		}
		for _, k := range x.Keywords {
			out = append(out, collectNamesForParCaptureExpr(k.Val)...)
		}
		return out
	case *parser.Name:
		return []*parser.Name{x}
	case *parser.Attribute:
		return collectNamesForParCaptureExpr(x.Value)
	case *parser.Subscript:
		return append(append(collectNamesForParCaptureExpr(x.Value), collectNamesForParCaptureExpr(x.Slice)...))
	case *parser.BinOp:
		return append(append(collectNamesForParCaptureExpr(x.Left), collectNamesForParCaptureExpr(x.Right)...))
	case *parser.UnaryOp:
		return collectNamesForParCaptureExpr(x.Expr)
	case *parser.Compare:
		out := collectNamesForParCaptureExpr(x.Left)
		for _, c := range x.Comparators {
			out = append(out, collectNamesForParCaptureExpr(c)...)
		}
		return out
	case *parser.BoolOp:
		var out []*parser.Name
		for _, v := range x.Values {
			out = append(out, collectNamesForParCaptureExpr(v)...)
		}
		return out
	case *parser.IfExp:
		return append(append(append(
			collectNamesForParCaptureExpr(x.Test),
			collectNamesForParCaptureExpr(x.Body)...),
			collectNamesForParCaptureExpr(x.Orelse)...))
	case *parser.ListExpr:
		var out []*parser.Name
		for _, el := range x.Elts {
			out = append(out, collectNamesForParCaptureExpr(el)...)
		}
		return out
	case *parser.TupleExpr:
		var out []*parser.Name
		for _, el := range x.Elts {
			out = append(out, collectNamesForParCaptureExpr(el)...)
		}
		return out
	case *parser.ListComp:
		out := collectNamesForParCaptureExpr(x.Elt)
		for _, c := range x.Comps {
			out = append(out, collectNamesForParCaptureExpr(c.Target)...)
			out = append(out, collectNamesForParCaptureExpr(c.Iter)...)
			for _, cond := range c.Ifs {
				out = append(out, collectNamesForParCaptureExpr(cond)...)
			}
		}
		return out
	case *parser.Starred:
		return collectNamesForParCaptureExpr(x.Value)
	default:
		return nil
	}
}

func collectNamesInStmts(stmts []parser.Stmt) []*parser.Name {
	var out []*parser.Name
	for _, st := range stmts {
		out = append(out, collectNamesInStmt(st)...)
	}
	return out
}

func collectNamesInStmt(st parser.Stmt) []*parser.Name {
	var out []*parser.Name
	switch s := st.(type) {
	case *parser.AssignStmt:
		for _, t := range s.Targets {
			out = append(out, collectNamesInExpr(t)...)
		}
		out = append(out, collectNamesInExpr(s.Value)...)
	case *parser.AnnAssign:
		out = append(out, collectNamesInExpr(s.Target)...)
		out = append(out, collectNamesInExpr(s.Annotation)...)
		if s.Value != nil {
			out = append(out, collectNamesInExpr(s.Value)...)
		}
	case *parser.ExprStmt:
		out = append(out, collectNamesInExpr(s.Value)...)
	case *parser.ReturnStmt:
		for _, e := range s.Values {
			out = append(out, collectNamesInExpr(e)...)
		}
	case *parser.IfStmt:
		out = append(out, collectNamesInExpr(s.Test)...)
		for _, x := range s.Body {
			out = append(out, collectNamesInStmt(x)...)
		}
		for _, e := range s.Elifs {
			out = append(out, collectNamesInExpr(e.Test)...)
			for _, x := range e.Body {
				out = append(out, collectNamesInStmt(x)...)
			}
		}
		for _, x := range s.Else {
			out = append(out, collectNamesInStmt(x)...)
		}
	case *parser.WhileStmt:
		out = append(out, collectNamesInExpr(s.Test)...)
		for _, x := range s.Body {
			out = append(out, collectNamesInStmt(x)...)
		}
	case *parser.ForStmt:
		out = append(out, collectNamesInExpr(s.Iter)...)
		out = append(out, collectNamesInExpr(s.Target)...)
		out = append(out, collectNamesInStmts(s.Body)...)
	case *parser.ParForStmt:
		out = append(out, collectNamesInExpr(s.Iter)...)
		out = append(out, collectNamesInExpr(s.Target)...)
		out = append(out, collectNamesInStmts(s.Body)...)
	case *parser.FunctionDef:
		for _, x := range s.Body {
			out = append(out, collectNamesInStmt(x)...)
		}
	case *parser.ClassDef:
		for _, x := range s.Body {
			out = append(out, collectNamesInStmt(x)...)
		}
	case *parser.UnsafeStmt:
		out = append(out, collectNamesInStmts(s.Body)...)
	}
	return out
}

func collectNamesInExpr(e parser.Expr) []*parser.Name {
	if e == nil {
		return nil
	}
	var out []*parser.Name
	switch x := e.(type) {
	case *parser.Name:
		return []*parser.Name{x}
	case *parser.Call:
		out = append(out, collectNamesInExpr(x.Func)...)
		for _, a := range x.Args {
			out = append(out, collectNamesInExpr(a)...)
		}
		for _, k := range x.Keywords {
			out = append(out, collectNamesInExpr(k.Val)...)
		}
	case *parser.Attribute:
		out = append(out, collectNamesInExpr(x.Value)...)
	case *parser.Subscript:
		out = append(out, collectNamesInExpr(x.Value)...)
		out = append(out, collectNamesInExpr(x.Slice)...)
	case *parser.BinOp:
		out = append(out, collectNamesInExpr(x.Left)...)
		out = append(out, collectNamesInExpr(x.Right)...)
	case *parser.UnaryOp:
		out = append(out, collectNamesInExpr(x.Expr)...)
	case *parser.Compare:
		out = append(out, collectNamesInExpr(x.Left)...)
		for _, c := range x.Comparators {
			out = append(out, collectNamesInExpr(c)...)
		}
	case *parser.BoolOp:
		for _, v := range x.Values {
			out = append(out, collectNamesInExpr(v)...)
		}
	case *parser.IfExp:
		out = append(out, collectNamesInExpr(x.Test)...)
		out = append(out, collectNamesInExpr(x.Body)...)
		out = append(out, collectNamesInExpr(x.Orelse)...)
	case *parser.ListExpr:
		for _, el := range x.Elts {
			out = append(out, collectNamesInExpr(el)...)
		}
	case *parser.TupleExpr:
		for _, el := range x.Elts {
			out = append(out, collectNamesInExpr(el)...)
		}
	case *parser.ListComp:
		out = append(out, collectNamesInExpr(x.Elt)...)
		for _, c := range x.Comps {
			out = append(out, collectNamesInExpr(c.Target)...)
			out = append(out, collectNamesInExpr(c.Iter)...)
			for _, cond := range c.Ifs {
				out = append(out, collectNamesInExpr(cond)...)
			}
		}
	case *parser.Starred:
		out = append(out, collectNamesInExpr(x.Value)...)
	}
	return out
}

type ownMode int

const (
	discardMode ownMode = iota
	consumeMode
)

func (p *opass) resetLinearSlotTarget(t parser.Expr) {
	at, ok := t.(*parser.Attribute)
	if !ok || !IsLinearType(ifaceTypeFromExpr(at)) {
		return
	}
	nm, ok := at.Value.(*parser.Name)
	if !ok {
		return
	}
	sym, ok := nm.Sym.(*Symbol)
	if !ok || sym == nil {
		return
	}
	delete(p.linearSlots, linearSlotKey{base: sym, field: at.Attr})
}

func (p *opass) useLinearAttr(at *parser.Attribute, mode ownMode) {
	if !IsLinearType(ifaceTypeFromExpr(at)) {
		return
	}
	nm, ok := at.Value.(*parser.Name)
	if !ok {
		return
	}
	sym, ok := nm.Sym.(*Symbol)
	if !ok || sym == nil {
		return
	}
	key := linearSlotKey{base: sym, field: at.Attr}
	line, col := nm.Line, nm.Col
	if mode == discardMode {
		return
	}
	if _, gone := p.linearSlots[key]; gone {
		p.errf(line, col, "UseAfterMoveError: `soa` field %q on %q was already moved or consumed", at.Attr, nm.Id)
		return
	}
	p.linearSlots[key] = struct{}{}
}

func (p *opass) ownExpr(e parser.Expr, mode ownMode) {
	if e == nil {
		return
	}
	switch x := e.(type) {
	case *parser.Name:
		p.useName(x, mode, false)
	case *parser.Call:
		if fn, ok := x.Func.(*parser.Name); ok {
			switch fn.Id {
			case "borrow":
				if len(x.Args) == 1 {
					p.ownBorrowArg(x.Args[0])
					return
				}
			case "timetravel_borrow":
				if len(x.Args) == 1 {
					p.ownTimeTravelBorrowArg(x.Args[0])
					return
				}
			case "mutborrow":
				if len(x.Args) == 1 {
					p.ownMutBorrowArg(x.Args[0])
					return
				}
			}
		}
		p.ownExpr(x.Func, discardMode)
		for _, a := range x.Args {
			p.ownExpr(a, consumeMode)
		}
		for _, k := range x.Keywords {
			p.ownExpr(k.Val, consumeMode)
		}
	case *parser.Attribute:
		if IsLinearType(ifaceTypeFromExpr(x)) {
			p.useLinearAttr(x, mode)
			p.ownExpr(x.Value, discardMode)
			return
		}
		p.ownExpr(x.Value, mode)
	case *parser.Subscript:
		p.ownExpr(x.Value, mode)
		p.ownExpr(x.Slice, discardMode)
	case *parser.BinOp:
		p.ownExpr(x.Left, consumeMode)
		p.ownExpr(x.Right, consumeMode)
	case *parser.UnaryOp:
		p.ownExpr(x.Expr, mode)
	case *parser.Compare:
		p.ownExpr(x.Left, discardMode)
		for _, c := range x.Comparators {
			p.ownExpr(c, discardMode)
		}
	case *parser.BoolOp:
		for _, v := range x.Values {
			p.ownExpr(v, discardMode)
		}
	case *parser.IfExp:
		p.ownExpr(x.Test, discardMode)
		p.ownExpr(x.Body, mode)
		p.ownExpr(x.Orelse, mode)
	case *parser.ListExpr:
		for _, el := range x.Elts {
			p.ownExpr(el, consumeMode)
		}
	case *parser.TupleExpr:
		for _, el := range x.Elts {
			p.ownExpr(el, consumeMode)
		}
	case *parser.ListComp:
		p.ownExpr(x.Elt, consumeMode)
		for _, c := range x.Comps {
			p.ownExpr(c.Target, discardMode)
			p.ownExpr(c.Iter, discardMode)
			for _, cond := range c.Ifs {
				p.ownExpr(cond, discardMode)
			}
		}
	case *parser.Starred:
		p.ownExpr(x.Value, consumeMode)
	default:
	}
}

func (p *opass) borrowRegionNote() string {
	fr := p.curBorrowFrame()
	if fr.tEpoch <= 0 {
		return ""
	}
	return fmt.Sprintf(" [spacetime region t=%d]", fr.tEpoch)
}

func (p *opass) ownBorrowArg(e parser.Expr) {
	nm, ok := e.(*parser.Name)
	if !ok {
		p.ownExpr(e, discardMode)
		return
	}
	sym, ok := nm.Sym.(*Symbol)
	if !ok || sym == nil {
		return
	}
	st := p.get(sym)
	if st.mutActive {
		p.errf(nm.Line, nm.Col, "BorrowError: cannot borrow %q while a mutable borrow (mutborrow) is active%s", nm.Id, p.borrowRegionNote())
		return
	}
	st.immBorrows++
	fr := p.curBorrowFrame()
	fr.imm = append(fr.imm, sym)
}

func (p *opass) ownTimeTravelBorrowArg(e parser.Expr) {
	// Ownership rules match borrow(...); MIR distinguishes time-travel for codegen (Pass 7).
	p.ownBorrowArg(e)
}

func (p *opass) ownMutBorrowArg(e parser.Expr) {
	nm, ok := e.(*parser.Name)
	if !ok {
		p.ownExpr(e, discardMode)
		return
	}
	sym, ok := nm.Sym.(*Symbol)
	if !ok || sym == nil {
		return
	}
	st := p.get(sym)
	if st.immBorrows > 0 {
		p.errf(nm.Line, nm.Col, "BorrowError: cannot mutborrow %q while immutable borrows are active (reborrow conflict)%s", nm.Id, p.borrowRegionNote())
		return
	}
	if st.mutActive {
		p.errf(nm.Line, nm.Col, "BorrowError: double mutable borrow of %q in the same region%s", nm.Id, p.borrowRegionNote())
		return
	}
	st.mutActive = true
	fr := p.curBorrowFrame()
	fr.mut = append(fr.mut, sym)
}

func (p *opass) useName(n *parser.Name, mode ownMode, _ bool) {
	sym, ok := n.Sym.(*Symbol)
	if !ok || sym == nil || sym.Kind == SymBuiltin {
		return
	}
	st := p.get(sym)
	if st.mutActive && mode == consumeMode {
		p.errf(n.Line, n.Col, "UseError: cannot use %q while mutborrow is active — finish the mutable borrow region first", n.Id)
		return
	}
	if IsLinearType(exprTypeFromAST(n)) {
		if st.immBorrows > 0 && mode == consumeMode {
			p.errf(n.Line, n.Col, "BorrowError: cannot move or consume linear value %q while borrow(...) is active — drop borrows first", n.Id)
			return
		}
		if st.moved && mode == consumeMode {
			p.errf(n.Line, n.Col, "UseAfterMoveError: %q was already moved or consumed (linear / soa value)", n.Id)
			return
		}
		if mode == consumeMode {
			st.moved = true
		}
	}
}

func (p *opass) taintPythonNames(e parser.Expr) {
	for _, n := range collectNamesInExpr(e) {
		if sym, ok := n.Sym.(*Symbol); ok && sym != nil {
			sym.TaintedPython = true
		}
	}
}

func (p *opass) exprHasTainted(e parser.Expr) bool {
	for _, n := range collectNamesInExpr(e) {
		if sym, ok := n.Sym.(*Symbol); ok && sym != nil && sym.TaintedPython {
			return true
		}
	}
	return false
}

func ifaceTypeFromExpr(e parser.Expr) Type {
	switch x := e.(type) {
	case *parser.Name:
		if t, ok := x.Typ.(Type); ok {
			return t
		}
	case *parser.Attribute:
		if t, ok := x.Typ.(Type); ok {
			return t
		}
	case *parser.Subscript:
		if t, ok := x.Typ.(Type); ok {
			return t
		}
	}
	return nil
}
