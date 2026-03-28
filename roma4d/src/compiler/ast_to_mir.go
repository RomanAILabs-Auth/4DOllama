package compiler

import (
	"fmt"
	"strconv"

	"github.com/RomanAILabs-Auth/Roma4D/src/parser"
)

// LowerToMIR builds a MIR module from a type-checked, ownership-checked unit.
// If cr.Errors is non-empty, it returns nil and a copy of those diagnostics.
func LowerToMIR(cr *CheckResult) (*MIRModule, []string) {
	if cr == nil {
		return nil, []string{"LowerToMIR: nil CheckResult"}
	}
	if len(cr.Errors) > 0 {
		out := make([]string, len(cr.Errors))
		copy(out, cr.Errors)
		return nil, out
	}
	lw := &mirLower{
		mod: &MIRModule{
			SourcePath: cr.Module.Filename,
			Qual:       cr.Module.Qual,
		},
	}
	lw.lowerModule(cr.Module)
	return lw.mod, lw.errs
}

type mirLower struct {
	mod    *MIRModule
	fn     *MIRFunction
	block  *MIRBlock
	nextID MIRValueID
	locals map[string]MIRValueID
	errs   []string

	// Compile-time spacetime region stack (Pass 8); encoded in MIR ImmI only, never at runtime.
	ctRegionNext  int64
	ctRegionStack []int64
}

func (lw *mirLower) pushCTRegion() int64 {
	lw.ctRegionNext++
	rid := lw.ctRegionNext
	lw.ctRegionStack = append(lw.ctRegionStack, rid)
	return rid
}

func (lw *mirLower) popCTRegion() {
	if n := len(lw.ctRegionStack); n > 0 {
		lw.ctRegionStack = lw.ctRegionStack[:n-1]
	}
}

func (lw *mirLower) activeCTRegion() int64 {
	if n := len(lw.ctRegionStack); n > 0 {
		return lw.ctRegionStack[n-1]
	}
	return 0
}

func (lw *mirLower) newValue() MIRValueID {
	lw.nextID++
	return lw.nextID
}

func (lw *mirLower) emit(ins MIRInst) {
	if lw.block == nil {
		return
	}
	lw.block.Insts = append(lw.block.Insts, ins)
}

func metaFromName(n *parser.Name) OwnershipMeta {
	sym, _ := n.Sym.(*Symbol)
	if sym == nil {
		return OwnershipMeta{}
	}
	m := OwnershipMeta{
		Sendable:      sym.Sendable,
		PythonTainted: sym.TaintedPython,
	}
	if sym.Type != nil {
		m.IsLinear = IsLinearType(sym.Type)
	}
	return m
}

func (lw *mirLower) lowerModule(m *parser.Module) {
	var other []parser.Stmt
	for _, st := range m.Body {
		switch s := st.(type) {
		case *parser.ClassDef:
			lw.lowerClassDecl(s)
		case *parser.FunctionDef:
			lw.lowerFunction(s)
		default:
			other = append(other, st)
		}
	}
	if len(other) == 0 {
		return
	}
	fn := &MIRFunction{Name: "__toplevel__", Blocks: []MIRBlock{{Label: "entry", Insts: nil}}}
	lw.fn = fn
	lw.block = &fn.Blocks[0]
	lw.locals = make(map[string]MIRValueID)
	lw.nextID = 0
	for _, st := range other {
		lw.emit(MIRInst{Kind: MIRComment, ImmS: fmt.Sprintf("top-level: %T", st)})
	}
	lw.mod.Funcs = append(lw.mod.Funcs, fn)
}

func (lw *mirLower) lowerClassDecl(cd *parser.ClassDef) {
	var soa []string
	if sym, ok := cd.Sym.(*Symbol); ok && sym != nil {
		if cls, ok := sym.Type.(*Class); ok && cls != nil {
			for fname, f := range cls.Fields {
				if f != nil && f.Layout == LayoutSoa {
					soa = append(soa, fname)
				}
			}
		}
	}
	lw.mod.Classes = append(lw.mod.Classes, MIRClassDecl{Name: cd.Name, SoaFields: soa})
}

func (lw *mirLower) lowerFunction(fd *parser.FunctionDef) {
	fn := &MIRFunction{
		Name:       fd.Name,
		QualModule: lw.mod.Qual,
		Blocks:     []MIRBlock{{Label: "entry", Insts: nil}},
	}
	lw.fn = fn
	lw.block = &fn.Blocks[0]
	lw.locals = make(map[string]MIRValueID)
	lw.nextID = 0
	lw.ctRegionNext = 0
	lw.ctRegionStack = nil

	for _, a := range fd.Args.Args {
		id := lw.newValue()
		p := MIRParam{Name: a.Name, Ty: MIRTypeRef{Name: "any"}}
		if a.Annot != nil {
			p.Ty = mirTypeRefFromType(typeFromAnnotExpr(lw, a.Annot))
		}
		fn.Params = append(fn.Params, p)
		lw.locals[a.Name] = id
		fn.Locals = append(fn.Locals, a.Name)
		lw.emit(MIRInst{Kind: MIRComment, ImmS: fmt.Sprintf("ssa_param %s %s : %s", a.Name, id.String(), p.Ty.Name)})
	}

	for _, st := range fd.Body {
		lw.lowerStmt(st)
	}
	lw.mod.Funcs = append(lw.mod.Funcs, fn)
}

func typeFromAnnotExpr(lw *mirLower, e parser.Expr) Type {
	switch n := e.(type) {
	case *parser.Name:
		return typeFromNameStatic(n.Id)
	case *parser.Subscript:
		if nm, ok := n.Value.(*parser.Name); ok && nm.Id == "list" {
			return &List{Elem: typeFromAnnotExpr(lw, n.Slice)}
		}
	}
	return TypAny
}

func typeFromNameStatic(id string) Type {
	switch id {
	case "int":
		return TypInt
	case "float":
		return TypFloat
	case "str":
		return TypStr
	case "bool":
		return TypBool
	case "vec4":
		return TypVec4
	case "rotor":
		return TypRotor
	case "multivector":
		return TypMultivector
	case "rawptr":
		return TypRawPtr
	}
	return TypAny
}

func (lw *mirLower) lowerStmt(st parser.Stmt) {
	switch s := st.(type) {
	case *parser.ReturnStmt:
		var uses []MIRValueID
		for _, e := range s.Values {
			uses = append(uses, lw.lowerExpr(e))
		}
		lw.emit(MIRInst{Kind: MIRReturn, Uses: uses})
	case *parser.AssignStmt:
		v := lw.lowerExpr(s.Value)
		for _, t := range s.Targets {
			lw.lowerAssignTarget(t, v)
		}
	case *parser.AnnAssign:
		var v MIRValueID
		if s.Value != nil {
			v = lw.lowerExpr(s.Value)
		}
		if nm, ok := s.Target.(*parser.Name); ok {
			lw.locals[nm.Id] = v
			lw.fn.Locals = append(lw.fn.Locals, nm.Id)
			ty := mirTypeRefFromType(typeFromAnnotExpr(lw, s.Annotation))
			if t, ok := nm.Typ.(Type); ok && t != nil {
				ty = mirTypeRefFromType(t)
			}
			lw.emit(MIRInst{Kind: MIRStoreLocal, Uses: []MIRValueID{v}, Name: nm.Id, Ty: ty, Own: metaFromName(nm)})
		} else {
			lw.lowerAssignTarget(s.Target, v)
		}
	case *parser.ExprStmt:
		_ = lw.lowerExpr(s.Value)
	case *parser.PassStmt:
		lw.emit(MIRInst{Kind: MIRNop})
	case *parser.ParForStmt:
		lw.lowerParFor(s)
	case *parser.UnsafeStmt:
		var kids []MIRInst
		saved := lw.block
		blk := &MIRBlock{Label: "unsafe_tmp", Insts: nil}
		lw.block = blk
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
		kids = append(kids, blk.Insts...)
		lw.block = saved
		lw.emit(MIRInst{Kind: MIRUnsafeRegion, Children: kids})
	case *parser.SpacetimeStmt:
		rid := lw.pushCTRegion()
		var kids []MIRInst
		saved := lw.block
		blk := &MIRBlock{Label: "spacetime_tmp", Insts: nil}
		lw.block = blk
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
		kids = append(kids, blk.Insts...)
		lw.block = saved
		lw.popCTRegion()
		lw.emit(MIRInst{Kind: MIRSpacetimeRegion, Children: kids, ImmI: rid, ImmS: "spacetime"})
	case *parser.IfStmt:
		if cmp, ok := s.Test.(*parser.Compare); ok && len(cmp.Ops) == 1 && len(cmp.Comparators) == 1 && cmp.Ops[0] == parser.EQEQ &&
			len(s.Elifs) == 0 {
			l := lw.lowerExpr(cmp.Left)
			r := lw.lowerExpr(cmp.Comparators[0])
			save := lw.block
			tb := &MIRBlock{Label: "if_then_tmp", Insts: nil}
			lw.block = tb
			for _, x := range s.Body {
				lw.lowerStmt(x)
			}
			thenKids := append([]MIRInst(nil), tb.Insts...)
			eb := &MIRBlock{Label: "if_else_tmp", Insts: nil}
			lw.block = eb
			for _, x := range s.Else {
				lw.lowerStmt(x)
			}
			elseKids := append([]MIRInst(nil), eb.Insts...)
			lw.block = save
			lw.emit(MIRInst{Kind: MIRIfStrEq, Uses: []MIRValueID{l, r}, Children: thenKids, AltChildren: elseKids})
			break
		}
		lw.emit(MIRInst{Kind: MIRComment, ImmS: "if: lowered body (v0 linear scan)"})
		lw.emit(MIRInst{Kind: MIRComment, ImmS: "then"})
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
		for _, e := range s.Elifs {
			_ = lw.lowerExpr(e.Test)
			for _, x := range e.Body {
				lw.lowerStmt(x)
			}
		}
		for _, x := range s.Else {
			lw.lowerStmt(x)
		}
	case *parser.WhileStmt:
		_ = lw.lowerExpr(s.Test)
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
	case *parser.ForStmt:
		if lw.lowerForStmtUnrolled(s) {
			break
		}
		_ = lw.lowerExpr(s.Iter)
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
	default:
		lw.emit(MIRInst{Kind: MIRComment, ImmS: fmt.Sprintf("stmt %T not lowered", st)})
	}
}

// forRangeUnrollMax caps `for x in range(N)` compile-time unrolling (guards IR size).
const forRangeUnrollMax = 512

// lowerForStmtUnrolled expands `for x in range(N):` with constant N into N sequential bodies
// with x bound to 0..N-1 (avoids single-shot body + undefined loop index).
func (lw *mirLower) lowerForStmtUnrolled(s *parser.ForStmt) bool {
	n, ok := forRangeConstN(s.Iter)
	if !ok {
		return false
	}
	for k := 0; k < n; k++ {
		id := lw.newValue()
		lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: int64(k), Ty: MIRTypeRef{Name: "int"}})
		lw.lowerAssignTarget(s.Target, id)
		for _, x := range s.Body {
			lw.lowerStmt(x)
		}
	}
	return true
}

func forRangeConstN(it parser.Expr) (int, bool) {
	call, ok := it.(*parser.Call)
	if !ok {
		return 0, false
	}
	fn, ok := call.Func.(*parser.Name)
	if !ok || fn.Id != "range" {
		return 0, false
	}
	if len(call.Args) != 1 || len(call.Keywords) > 0 {
		return 0, false
	}
	c, ok := call.Args[0].(*parser.Constant)
	if !ok || c.Kind != parser.INT {
		return 0, false
	}
	v, err := strconv.ParseInt(c.Text, 0, 64)
	if err != nil || v < 0 || v > int64(forRangeUnrollMax) {
		return 0, false
	}
	return int(v), true
}

func (lw *mirLower) lowerParFor(s *parser.ParForStmt) {
	lv := ""
	if nm, ok := s.Target.(*parser.Name); ok {
		lv = nm.Id
	}
	_ = lw.lowerExpr(s.Iter)
	saved := lw.block
	blk := &MIRBlock{Label: "par_tmp", Insts: nil}
	lw.block = blk
	for _, x := range s.Body {
		lw.lowerStmt(x)
	}
	kids := append([]MIRInst(nil), blk.Insts...)
	lw.block = saved
	extra := ""
	if len(lw.ctRegionStack) > 0 {
		extra = "gpu_spacetime_par"
	}
	lw.emit(MIRInst{Kind: MIRParRegion, Name: lv, Extra: extra, Children: kids})
}

func (lw *mirLower) lowerAssignTarget(t parser.Expr, v MIRValueID) {
	switch n := t.(type) {
	case *parser.Name:
		lw.locals[n.Id] = v
		lw.fn.Locals = append(lw.fn.Locals, n.Id)
		ty := mirTypeRefFromType(anyTypeFromExpr(n))
		lw.emit(MIRInst{Kind: MIRStoreLocal, Uses: []MIRValueID{v}, Name: n.Id, Ty: ty, Own: metaFromName(n)})
	case *parser.Attribute:
		base := lw.lowerExpr(n.Value)
		tty, _ := n.Typ.(Type)
		lw.emit(MIRInst{
			Kind: MIRSoaStore,
			Uses: []MIRValueID{base, v},
			ImmS: n.Attr,
			Ty:   mirTypeRefFromType(tty),
			Own:  OwnershipMeta{IsLinear: IsLinearType(tty)},
		})
	case *parser.Subscript:
		obj := lw.lowerExpr(n.Value)
		key := lw.lowerExpr(n.Slice)
		lw.emit(MIRInst{Kind: MIRBinOp, Name: "setitem", Uses: []MIRValueID{obj, key, v}})
	default:
		lw.emit(MIRInst{Kind: MIRComment, ImmS: fmt.Sprintf("assign target %T", t)})
	}
}

func anyTypeFromExpr(e parser.Expr) Type {
	switch x := e.(type) {
	case *parser.Name:
		if t, ok := x.Typ.(Type); ok {
			return t
		}
	}
	return TypAny
}

func (lw *mirLower) lowerExpr(e parser.Expr) MIRValueID {
	if e == nil {
		return 0
	}
	switch x := e.(type) {
	case *parser.TimeCoord:
		id := lw.newValue()
		lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: 0, Ty: MIRTypeRef{Name: "time", TimeTag: "current"}})
		return id
	case *parser.Constant:
		switch x.Kind {
		case parser.INT:
			v, _ := strconv.ParseInt(x.Text, 0, 64)
			id := lw.newValue()
			lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: v})
			return id
		case parser.FLOAT:
			f, _ := strconv.ParseFloat(x.Text, 64)
			id := lw.newValue()
			lw.emit(MIRInst{Kind: MIRConstFloat, Dst: id, ImmF: f})
			return id
		case parser.STRING:
			id := lw.newValue()
			lw.emit(MIRInst{Kind: MIRConstStr, Dst: id, ImmS: x.Text})
			return id
		case parser.KwTrue, parser.KwFalse:
			id := lw.newValue()
			var v int64
			if x.Kind == parser.KwTrue {
				v = 1
			}
			lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: v, Ty: MIRTypeRef{Name: "bool"}})
			return id
		case parser.KwNone:
			id := lw.newValue()
			lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: 0, Ty: MIRTypeRef{Name: "none"}})
			return id
		}
	case *parser.Name:
		if id, ok := lw.locals[x.Id]; ok {
			return id
		}
		out := lw.newValue()
		ty := mirTypeRefFromType(anyTypeFromExpr(x))
		lw.emit(MIRInst{Kind: MIRLoadLocal, Dst: out, Name: x.Id, Ty: ty, Own: metaFromName(x)})
		return out
	case *parser.Call:
		return lw.lowerCall(x)
	case *parser.BinOp:
		if x.Op == parser.AT {
			base := lw.lowerExpr(x.Left)
			tcoord := lw.lowerExpr(x.Right)
			dst := lw.newValue()
			ty := mirTypeRefFromType(exprASTType(x))
			lw.emit(MIRInst{
				Kind: MIRTemporalMove,
				Dst:  dst,
				Uses: []MIRValueID{base, tcoord},
				Ty:   ty,
				ImmS: "t",
				ImmI: lw.activeCTRegion(),
				Own:  OwnershipMeta{Notes: "spacetime@t"},
			})
			return dst
		}
		if x.Op == parser.STAR {
			lt := exprASTType(x.Left)
			rt := exprASTType(x.Right)
			if isGA(lt) || isGA(rt) {
				a, b := lw.lowerExpr(x.Left), lw.lowerExpr(x.Right)
				dst := lw.newValue()
				lw.emit(MIRInst{Kind: MIRGeomMul, Dst: dst, Uses: []MIRValueID{a, b}})
				return dst
			}
		}
		if x.Op == parser.CARET {
			lt := exprASTType(x.Left)
			rt := exprASTType(x.Right)
			if isGA(lt) || isGA(rt) {
				a, b := lw.lowerExpr(x.Left), lw.lowerExpr(x.Right)
				dst := lw.newValue()
				lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{a, b}, Name: "ga_xor"})
				return dst
			}
		}
		if x.Op == parser.PIPE {
			lt := exprASTType(x.Left)
			rt := exprASTType(x.Right)
			if isGA(lt) || isGA(rt) {
				a, b := lw.lowerExpr(x.Left), lw.lowerExpr(x.Right)
				dst := lw.newValue()
				lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{a, b}, Name: "ga_inner"})
				return dst
			}
		}
		a, b := lw.lowerExpr(x.Left), lw.lowerExpr(x.Right)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{a, b}, Name: tokenOpName(x.Op)})
		return dst
	case *parser.UnaryOp:
		v := lw.lowerExpr(x.Expr)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{v}, Name: "unary"})
		return dst
	case *parser.Subscript:
		o := lw.lowerExpr(x.Value)
		k := lw.lowerExpr(x.Slice)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{o, k}, Name: "getitem"})
		return dst
	case *parser.Attribute:
		base := lw.lowerExpr(x.Value)
		tty, _ := x.Typ.(Type)
		dst := lw.newValue()
		if IsLinearType(tty) {
			lw.emit(MIRInst{
				Kind: MIRSoaLoad,
				Dst:  dst,
				Uses: []MIRValueID{base},
				ImmS: x.Attr,
				Ty:   mirTypeRefFromType(tty),
				Own:  OwnershipMeta{IsLinear: true, Sendable: false},
			})
			return dst
		}
		lw.emit(MIRInst{Kind: MIRBinOp, Dst: dst, Uses: []MIRValueID{base}, Name: "getattr", ImmS: x.Attr})
		return dst
	case *parser.ListComp:
		id := lw.newValue()
		lw.emit(MIRInst{Kind: MIRComment, ImmS: "list comprehension elided in MIR v0"})
		lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: 0})
		return id
	case *parser.ListExpr:
		id := lw.newValue()
		lw.emit(MIRInst{Kind: MIRComment, ImmS: "list literal elided in MIR v0"})
		lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: 0})
		return id
	case *parser.TupleExpr:
		if len(x.Elts) == 0 {
			id := lw.newValue()
			lw.emit(MIRInst{Kind: MIRConstInt, Dst: id, ImmI: 0, Ty: MIRTypeRef{Name: "tuple[]"}})
			return id
		}
		var last MIRValueID
		for _, el := range x.Elts {
			last = lw.lowerExpr(el)
		}
		id := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCopy, Dst: id, Uses: []MIRValueID{last}, Name: "tuple_tail"})
		return id
	}
	id := lw.newValue()
	lw.emit(MIRInst{Kind: MIRComment, ImmS: fmt.Sprintf("expr %T", e)})
	return id
}

func (lw *mirLower) lowerCall(n *parser.Call) MIRValueID {
	fn, ok := n.Func.(*parser.Name)
	if !ok {
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "?indirect", Ty: mirTypeRefFromType(TypAny)})
		return dst
	}
	switch fn.Id {
	case "borrow":
		var v MIRValueID
		if len(n.Args) == 1 {
			v = lw.lowerExpr(n.Args[0])
			lw.emit(MIRInst{Kind: MIRBorrow, Uses: []MIRValueID{v}})
		}
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCopy, Dst: dst, Uses: []MIRValueID{v}, Name: "borrow_result"})
		return dst
	case "mutborrow":
		var v MIRValueID
		if len(n.Args) == 1 {
			v = lw.lowerExpr(n.Args[0])
			lw.emit(MIRInst{Kind: MIRMutBorrow, Uses: []MIRValueID{v}})
		}
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCopy, Dst: dst, Uses: []MIRValueID{v}, Name: "mutborrow_result"})
		return dst
	case "timetravel_borrow":
		var v MIRValueID
		if len(n.Args) == 1 {
			v = lw.lowerExpr(n.Args[0])
			lw.emit(MIRInst{Kind: MIRTimeTravelBorrow, Uses: []MIRValueID{v}, ImmS: "t"})
		}
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRChronoRead, Dst: dst, Uses: []MIRValueID{v}, ImmI: lw.activeCTRegion()})
		return dst
	case "mir_alloc":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRHeapAlloc, Dst: dst, Uses: args, Ty: MIRTypeRef{Name: "rawptr"}})
		return dst
	case "mir_ptr_load":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRPtrLoad, Dst: dst, Uses: args})
		return dst
	case "mir_ptr_store":
		args := lw.lowerArgs(n)
		lw.emit(MIRInst{Kind: MIRPtrStore, Uses: args})
		return 0
	case "mir_mmap_gguf":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_mmap_gguf", Ty: MIRTypeRef{Name: "rawptr"}})
		return dst
	case "mir_get_ollama_qwen_path":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_get_ollama_qwen_path", Ty: MIRTypeRef{Name: "str"}})
		return dst
	case "mir_romanai_gguf_path":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_gguf_path", Ty: MIRTypeRef{Name: "str"}})
		return dst
	case "mir_romanai_prompt":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_prompt", Ty: MIRTypeRef{Name: "str"}})
		return dst
	case "mir_romanai_cli_model_path":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_cli_model_path", Ty: MIRTypeRef{Name: "str"}})
		return dst
	case "mir_romanai_gguf_manifest_log":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_manifest_log", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_layout_load":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_layout_load", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_layout_free":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_gguf_layout_free", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_layout_tensor_count":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_gguf_layout_tensor_count", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_layout_tensor_vec4_slots":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_layout_tensor_vec4_slots", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_layout_tensor_rawptr":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_layout_tensor_rawptr", Ty: MIRTypeRef{Name: "rawptr"}})
		return dst
	case "mir_romanai_gguf_layout_tensor_ggml_type":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_layout_tensor_ggml_type", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_tensor_quant_bridge_prepare":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_gguf_tensor_quant_bridge_prepare", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_gguf_tensor_quant_bridge_ptr":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_romanai_gguf_tensor_quant_bridge_ptr", Ty: MIRTypeRef{Name: "rawptr"}})
		return dst
	case "mir_romanai_input_line":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_input_line", Ty: MIRTypeRef{Name: "str"}})
		return dst
	case "mir_romanai_decode_graph_step":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_decode_graph_step", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_lattice_coupling_step":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_lattice_coupling_step", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_romanai_vocab_print":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "mir_romanai_vocab_print", Ty: MIRTypeRef{Name: "none"}})
		return dst
	case "mir_qwen_chat_loop":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "mir_qwen_chat_loop", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "mir_cast_to_vec4_list":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRViewVec4List, Dst: dst, Uses: args, Ty: MIRTypeRef{Name: "list[vec4]"}})
		return dst
	case "ollama_demo":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "ollama_demo", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "quantum_server_demo":
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: nil, Name: "quantum_server_demo", Ty: MIRTypeRef{Name: "int"}})
		return dst
	case "print":
		args := lw.lowerArgs(n)
		dst := lw.newValue()
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: "print", Ty: MIRTypeRef{Name: "none"}})
		return dst
	case "range", "len", "vec4", "rotor", "multivector", "int", "float", "str", "bool":
		args := kwLowerArgs(lw, n)
		dst := lw.newValue()
		ty := TypAny
		if t, ok := n.Typ.(Type); ok {
			ty = t
		}
		lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: fn.Id, Ty: mirTypeRefFromType(ty)})
		return dst
	}
	args := kwLowerArgs(lw, n)
	dst := lw.newValue()
	ty := TypAny
	if t, ok := n.Typ.(Type); ok {
		ty = t
	}
	lw.emit(MIRInst{Kind: MIRCall, Dst: dst, Uses: args, Name: fn.Id, Ty: mirTypeRefFromType(ty)})
	return dst
}

func (lw *mirLower) lowerArgs(n *parser.Call) []MIRValueID {
	var out []MIRValueID
	for _, a := range n.Args {
		out = append(out, lw.lowerExpr(a))
	}
	return out
}

func kwLowerArgs(lw *mirLower, n *parser.Call) []MIRValueID {
	var out []MIRValueID
	for _, a := range n.Args {
		out = append(out, lw.lowerExpr(a))
	}
	for _, k := range n.Keywords {
		out = append(out, lw.lowerExpr(k.Val))
	}
	return out
}

func exprASTType(e parser.Expr) Type {
	switch v := e.(type) {
	case *parser.TimeCoord:
		return TypTime
	case *parser.Name:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	case *parser.Constant:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	case *parser.BinOp:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	case *parser.Subscript:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	case *parser.Attribute:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	case *parser.Call:
		if t, ok := v.Typ.(Type); ok {
			return t
		}
	}
	return TypAny
}

func tokenOpName(op parser.TokenKind) string {
	switch op {
	case parser.PLUS:
		return "add"
	case parser.MINUS:
		return "sub"
	case parser.STAR:
		return "mul"
	case parser.SLASH:
		return "div"
	case parser.AT:
		return "temporal_at"
	case parser.PERCENT:
		return "mod"
	case parser.CARET:
		return "xor_int"
	case parser.PIPE:
		return "or_int"
	default:
		return "binop"
	}
}
