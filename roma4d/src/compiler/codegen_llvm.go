package compiler

import (
	"fmt"
	"strings"

	"github.com/llir/llvm/ir"
	"github.com/llir/llvm/ir/constant"
	"github.com/llir/llvm/ir/enum"
	"github.com/llir/llvm/ir/types"
	"github.com/llir/llvm/ir/value"
)

// llvmGen lowers one MIR module to LLVM IR (llir) for CPU codegen (Pass 6).
type llvmGen struct {
	mod *ir.Module
	mir *MIRModule

	warn []string

	// className -> LLVM struct type (SoA fields as [4 x double] each, v0)
	classTys map[string]*types.StructType
	decls    map[string]*ir.Func

	// Per-function
	fn        *ir.Func
	block     *ir.Block
	vals      map[MIRValueID]value.Value
	slots     map[string]value.Value // current alloca per source local name
	slotTypes map[string]types.Type  // element type of each slot (detect shadowing / type change)
	retLLVM   types.Type             // LLVM return type for current fn
	features  mirFeatureFlags

	// Per-function uniquifier for named LLVM locals (allocas, geom temps, etc.).
	localCounter int

	// loweringFn is the MIR function currently being lowered (for MIRReturn typing).
	loweringFn *MIRFunction
}

type mirFeatureFlags struct {
	hasPar             bool
	hasUnsafe          bool
	hasSoa             bool
	hasGeom            bool
	hasSimdGeom        bool
	hasGPUParSpacetime bool
	hasSpacetime       bool
	hasTemporal        bool
	hasTimeTravel      bool
}

// LowerMIRToLLVM translates MIR to an LLVM IR module (in-memory). Warnings are non-fatal.
func LowerMIRToLLVM(m *MIRModule) (*ir.Module, []string, error) {
	if m == nil {
		return nil, nil, fmt.Errorf("LowerMIRToLLVM: nil MIRModule")
	}
	g := &llvmGen{
		mod:      ir.NewModule(),
		mir:      m,
		classTys: make(map[string]*types.StructType),
		decls:    make(map[string]*ir.Func),
	}
	g.mod.SourceFilename = m.SourcePath
	g.mod.TargetTriple = "" // host default when compiling .ll (zig cc or clang)

	g.registerClassLayouts()
	g.scanFeatures()

	for _, f := range m.Funcs {
		if f == nil || f.Name == "__toplevel__" {
			continue
		}
		if err := g.lowerFunc(f); err != nil {
			return nil, g.warn, err
		}
	}

	g.emitFeatureGlobals()
	g.appendAnnotationComments()

	return g.mod, g.warn, nil
}

// LLVMModuleString returns the textual LLVM IR for m.
func LLVMModuleString(m *ir.Module) (string, error) {
	if m == nil {
		return "", fmt.Errorf("nil module")
	}
	if err := m.AssignGlobalIDs(); err != nil {
		return "", err
	}
	return m.String(), nil
}

func (g *llvmGen) registerClassLayouts() {
	vec := types.NewArray(4, types.Double)
	for _, c := range g.mir.Classes {
		if c.Name == "" {
			continue
		}
		fields := make([]types.Type, 0, len(c.SoaFields))
		for range c.SoaFields {
			fields = append(fields, vec)
		}
		if len(fields) == 0 {
			continue
		}
		st := types.NewStruct(fields...)
		st.SetName("roma4d.class." + c.Name)
		g.classTys[c.Name] = st
		g.mod.TypeDefs = append(g.mod.TypeDefs, st)
	}
}

func (g *llvmGen) scanFeatures() {
	for _, f := range g.mir.Funcs {
		if f == nil {
			continue
		}
		for _, b := range f.Blocks {
			g.scanInsts(b.Insts)
		}
	}
}

func (g *llvmGen) scanInsts(list []MIRInst) {
	for i := range list {
		in := &list[i]
		switch in.Kind {
		case MIRParRegion:
			g.features.hasPar = true
			if in.Extra == "gpu_spacetime_par" {
				g.features.hasGPUParSpacetime = true
			}
			g.scanInsts(in.Children)
		case MIRUnsafeRegion:
			g.features.hasUnsafe = true
			g.scanInsts(in.Children)
		case MIRSoaLoad, MIRSoaStore:
			g.features.hasSoa = true
		case MIRGeomMul:
			g.features.hasGeom = true
			g.features.hasSimdGeom = true
		case MIRSpacetimeRegion:
			g.features.hasSpacetime = true
			g.scanInsts(in.Children)
		case MIRTemporalMove:
			g.features.hasTemporal = true
		case MIRTimeTravelBorrow:
			g.features.hasTimeTravel = true
		case MIRChronoRead:
			g.features.hasTimeTravel = true
		case MIRIfStrEq:
			g.scanInsts(in.Children)
			g.scanInsts(in.AltChildren)
		case MIRViewVec4List:
			// no nested scan
		default:
			g.scanInsts(in.Children)
		}
	}
}

func (g *llvmGen) emitFeatureGlobals() {
	// Test-visible markers for Pass 6 (stripped or replaced in release pipelines later).
	if g.features.hasSoa {
		newMarkerGlobal(g.mod, "roma4d.mir.has_soa", true)
	}
	if g.features.hasPar {
		newMarkerGlobal(g.mod, "roma4d.mir.has_par", true)
	}
	if g.features.hasUnsafe {
		newMarkerGlobal(g.mod, "roma4d.mir.has_unsafe", true)
	}
	if g.features.hasGeom {
		newMarkerGlobal(g.mod, "roma4d.mir.has_geom", true)
	}
	if g.features.hasSimdGeom {
		newMarkerGlobal(g.mod, "roma4d.mir.simd_geom", true)
	}
	if g.features.hasGPUParSpacetime {
		newMarkerGlobal(g.mod, "roma4d.mir.gpu_par_spacetime", true)
	}
	if g.features.hasSpacetime {
		newMarkerGlobal(g.mod, "roma4d.mir.has_spacetime_region", true)
	}
	if g.features.hasTemporal {
		newMarkerGlobal(g.mod, "roma4d.mir.has_temporal", true)
	}
	if g.features.hasTimeTravel {
		newMarkerGlobal(g.mod, "roma4d.mir.has_timetravel_borrow", true)
	}
}

func newMarkerGlobal(m *ir.Module, name string, v bool) {
	var c constant.Constant
	if v {
		c = constant.True
	} else {
		c = constant.False
	}
	m.NewGlobalDef(name, c)
}

func (g *llvmGen) appendAnnotationComments() {
	// Module.String() does not carry freeform comments; feature globals above anchor tests.
	// Spacetime is compile-time only: temporal MIR lowers to ordinary LLVM values (no chrono calls).
	_ = g.mod
}

func (g *llvmGen) warnf(format string, args ...interface{}) {
	g.warn = append(g.warn, fmt.Sprintf(format, args...))
}

// freshName returns a unique local name per function, e.g. geom_out_1, geom_out_2.
func (g *llvmGen) freshName(base string) string {
	g.localCounter++
	base = sanitizeIdent(base)
	if base == "" {
		base = "tmp"
	}
	return fmt.Sprintf("%s_%d", base, g.localCounter)
}

func (g *llvmGen) lowerFunc(mf *MIRFunction) error {
	g.loweringFn = mf
	defer func() { g.loweringFn = nil }()

	retTy, err := g.inferFuncReturnLLVM(mf)
	if err != nil {
		return err
	}
	g.retLLVM = retTy

	params := make([]*ir.Param, 0, len(mf.Params))
	for _, p := range mf.Params {
		pt := mirTypeRefToLLVM(p.Ty, g.classTys)
		params = append(params, ir.NewParam(sanitizeIdent(p.Name), pt))
	}

	name := mf.Name
	if name == "main" {
		// C ABI: always i32 @main. Roma4D `-> None` / `return None` lowers to ret i32 0.
		// Avoids invalid IR (e.g. define void @main + synthesized ret i64) when inference
		// and falloff paths disagree.
		if retTy != types.I32 {
			if retTy != types.Void {
				g.warnf("function %q: forcing i32 return for @main (was %s)", name, retTy)
			}
			retTy = types.I32
			g.retLLVM = types.I32
		}
	}

	g.fn = g.mod.NewFunc(sanitizeIdent(name), retTy, params...)
	g.localCounter = 0
	if len(mf.Blocks) == 0 {
		return fmt.Errorf("MIR function %q has no blocks", mf.Name)
	}
	g.block = g.fn.NewBlock("entry")
	g.vals = make(map[MIRValueID]value.Value)
	g.slots = make(map[string]value.Value)
	g.slotTypes = make(map[string]types.Type)

	// Shadow stack slots for parameters (MIR models params as SSA ids + alloca-like locals).
	for i, p := range mf.Params {
		if i >= len(g.fn.Params) {
			break
		}
		slot := g.ensureSlot(g.block, p.Name, g.fn.Params[i].Typ)
		g.block.NewStore(g.fn.Params[i], slot)
	}

	for _, bl := range mf.Blocks {
		for i := range bl.Insts {
			if err := g.lowerInst(&bl.Insts[i]); err != nil {
				return fmt.Errorf("%s:%s: %w", mf.Name, bl.Label, err)
			}
		}
	}

	if g.block.Term == nil {
		if retTy == types.Void {
			g.block.NewRet(nil)
		} else if retTy == types.I32 {
			g.block.NewRet(constant.NewInt(types.I32, 0))
		} else {
			g.block.NewRet(constant.NewInt(types.I64, 0))
		}
		g.warnf("function %q: synthesized return", mf.Name)
	}
	return nil
}

func (g *llvmGen) inferFuncReturnLLVM(mf *MIRFunction) (types.Type, error) {
	for _, b := range mf.Blocks {
		for i := range b.Insts {
			if b.Insts[i].Kind == MIRReturn {
				if len(b.Insts[i].Uses) == 0 {
					return types.Void, nil
				}
				// Infer from MIR type of first returned SSA id (approximate).
				id := b.Insts[i].Uses[0]
				// Scan backwards for dst typing — fallback i32 for main-friendly ABI.
				ty := g.findTypeOfSSA(mf, id)
				lt := mirTypeRefToLLVM(ty, g.classTys)
				if mf.Name == "main" && lt == types.I64 {
					return types.I32, nil
				}
				return lt, nil
			}
		}
	}
	if mf.Name == "main" {
		return types.I32, nil
	}
	return types.Void, nil
}

func (g *llvmGen) findTypeOfSSA(mf *MIRFunction, id MIRValueID) MIRTypeRef {
	for _, b := range mf.Blocks {
		for i := range b.Insts {
			in := &b.Insts[i]
			if in.Dst == id {
				return in.Ty
			}
		}
	}
	return MIRTypeRef{Name: "int"}
}

func (g *llvmGen) ensureSlot(b *ir.Block, name string, typ types.Type) value.Value {
	if s, ok := g.slots[name]; ok {
		if prev := g.slotTypes[name]; prev != nil {
			if prev.Equal(typ) {
				return s
			}
			g.warnf("local %q: slot type changed %s -> %s (fresh alloca)", name, prev, typ)
		}
	}
	a := b.NewAlloca(typ)
	a.SetName(g.freshName(sanitizeIdent(name)))
	g.slots[name] = a
	g.slotTypes[name] = typ
	return a
}

func (g *llvmGen) lowerInst(in *MIRInst) error {
	b := g.block
	switch in.Kind {
	case MIRNop, MIRComment:
		return nil
	case MIRConstInt:
		v := constant.NewInt(types.I64, in.ImmI)
		if in.Dst != 0 {
			g.vals[in.Dst] = v
		}
	case MIRConstFloat:
		v := constant.NewFloat(types.Double, in.ImmF)
		if in.Dst != 0 {
			g.vals[in.Dst] = v
		}
	case MIRConstStr:
		// Global string (null-terminated) + pointer as i8*
		s := in.ImmS + "\x00"
		arr := constant.NewCharArrayFromString(s)
		glob := g.mod.NewGlobalDef(fmt.Sprintf("str.%d", len(g.mod.Globals)), arr)
		z := constant.NewInt(types.I64, 0)
		gep := constant.NewGetElementPtr(arr.Type(), glob, z, z)
		if in.Dst != 0 {
			g.vals[in.Dst] = gep
		}
	case MIRStoreLocal:
		if len(in.Uses) < 1 {
			return fmt.Errorf("store_local %q: missing value", in.Name)
		}
		src := g.resolve(in.Uses[0])
		lt := mirTypeRefToLLVM(in.Ty, g.classTys)
		slot := g.ensureSlot(b, in.Name, lt)
		src = g.coerceForStore(b, src, lt)
		b.NewStore(src, slot)
	case MIRLoadLocal:
		lt := mirTypeRefToLLVM(in.Ty, g.classTys)
		slot, ok := g.slots[in.Name]
		if !ok {
			slot = g.ensureSlot(b, in.Name, lt)
		} else if st := g.slotTypes[in.Name]; st != nil && !st.Equal(lt) {
			g.warnf("load %q: MIR type %s does not match slot %s (loading as slot type)", in.Name, lt, st)
			lt = st
		}
		v := b.NewLoad(lt, slot)
		if in.Dst != 0 {
			g.vals[in.Dst] = v
		}
	case MIRCopy:
		if len(in.Uses) < 1 {
			return fmt.Errorf("copy: missing operand")
		}
		x := g.resolve(in.Uses[0])
		if in.Dst != 0 {
			g.vals[in.Dst] = x
		}
	case MIRBinOp:
		if in.Name == "setitem" && len(in.Uses) >= 3 {
			lst, idx, val := g.resolve(in.Uses[0]), g.resolve(in.Uses[1]), g.resolve(in.Uses[2])
			f := g.ensureDecl("roma4d_list_set_vec4", types.Void,
				ir.NewParam("lst", types.NewPointer(types.I8)),
				ir.NewParam("i", types.I64),
				ir.NewParam("v", types.NewPointer(types.Double)))
			ii := idx
			if !ii.Type().Equal(types.I64) {
				ii = b.NewSExt(idx, types.I64)
			}
			vp := g.pointerCast(b, val, types.NewPointer(types.Double))
			b.NewCall(f, g.pointerCast(b, lst, types.NewPointer(types.I8)), ii, vp)
			return nil
		}
		if len(in.Uses) < 2 {
			return fmt.Errorf("binop %s: missing operands", in.Name)
		}
		x, y := g.resolve(in.Uses[0]), g.resolve(in.Uses[1])
		var out value.Value
		switch in.Name {
		case "add":
			if x.Type().Equal(types.Double) || y.Type().Equal(types.Double) {
				if !x.Type().Equal(types.Double) {
					x = b.NewSIToFP(x, types.Double)
				}
				if !y.Type().Equal(types.Double) {
					y = b.NewSIToFP(y, types.Double)
				}
				out = b.NewFAdd(x, y)
			} else {
				out = b.NewAdd(x, y)
			}
		case "sub":
			if x.Type().Equal(types.Double) || y.Type().Equal(types.Double) {
				if !x.Type().Equal(types.Double) {
					x = b.NewSIToFP(x, types.Double)
				}
				if !y.Type().Equal(types.Double) {
					y = b.NewSIToFP(y, types.Double)
				}
				out = b.NewFSub(x, y)
			} else {
				out = b.NewSub(x, y)
			}
		case "mul":
			if x.Type().Equal(types.Double) || y.Type().Equal(types.Double) {
				if !x.Type().Equal(types.Double) {
					x = b.NewSIToFP(x, types.Double)
				}
				if !y.Type().Equal(types.Double) {
					y = b.NewSIToFP(y, types.Double)
				}
				out = b.NewFMul(x, y)
			} else {
				out = b.NewMul(x, y)
			}
		case "div", "sdiv":
			if x.Type().Equal(types.Double) || y.Type().Equal(types.Double) {
				if !x.Type().Equal(types.Double) {
					x = b.NewSIToFP(x, types.Double)
				}
				if !y.Type().Equal(types.Double) {
					y = b.NewSIToFP(y, types.Double)
				}
				out = b.NewFDiv(x, y)
			} else {
				out = b.NewSDiv(x, y)
			}
		case "xor_int":
			out = b.NewXor(x, y)
		case "or_int":
			out = b.NewOr(x, y)
		case "getitem":
			// v0: treat as opaque ptr load — i64 offset into ptr
			out = g.lowerGetItem(b, x, y)
		case "ga_xor", "ga_inner":
			// Pass 6 stub: fold pointer operands to i64 for a deterministic LLVM shape (Pass 7: real GA).
			vx := g.ptrToI64(b, x)
			vy := g.ptrToI64(b, y)
			out = b.NewXor(vx, vy)
		default:
			out = b.NewAdd(x, y)
			g.warnf("unknown binop %q, lowered as add", in.Name)
		}
		if in.Dst != 0 {
			g.vals[in.Dst] = out
		}
	case MIRCall:
		return g.lowerCall(in)
	case MIRGeomMul:
		if len(in.Uses) < 2 {
			return fmt.Errorf("geom_mul: missing operands")
		}
		// vec4/rotor are opaque i8*; list elements may be misaligned for <4 x double> load (Zig UBSAN).
		// Copy through aligned stack slots, then SIMD fmul (keeps markers + IR shape tests happy).
		v4 := types.NewVector(4, types.Double)
		arr4 := types.NewArray(4, types.Double)
		mc := g.ensureDecl("memcpy", types.Void,
			ir.NewParam("dst", types.NewPointer(types.I8)),
			ir.NewParam("src", types.NewPointer(types.I8)),
			ir.NewParam("n", types.I64))
		n32 := constant.NewInt(types.I64, 32)
		z0 := constant.NewInt(types.I32, 0)
		va := b.NewAlloca(arr4)
		va.SetName(g.freshName("geom_v_align"))
		ra := b.NewAlloca(arr4)
		ra.SetName(g.freshName("geom_r_align"))
		outA := b.NewAlloca(arr4)
		outA.SetName(g.freshName("geom_out"))
		vSrc := g.pointerCast(b, g.resolve(in.Uses[0]), types.NewPointer(types.I8))
		rSrc := g.pointerCast(b, g.resolve(in.Uses[1]), types.NewPointer(types.I8))
		b.NewCall(mc, b.NewBitCast(va, types.NewPointer(types.I8)), vSrc, n32)
		b.NewCall(mc, b.NewBitCast(ra, types.NewPointer(types.I8)), rSrc, n32)
		vvPtr := b.NewBitCast(va, types.NewPointer(v4))
		rrPtr := b.NewBitCast(ra, types.NewPointer(v4))
		vv := b.NewLoad(v4, vvPtr)
		rr := b.NewLoad(v4, rrPtr)
		prod := b.NewFMul(vv, rr)
		outVP := b.NewBitCast(outA, types.NewPointer(v4))
		b.NewStore(prod, outVP)
		gep := b.NewGetElementPtr(arr4, outA, z0, z0)
		vecI8 := g.pointerCast(b, gep, types.NewPointer(types.I8))
		if in.Dst != 0 {
			g.vals[in.Dst] = vecI8
		}
	case MIRReturn:
		// @main is always i32 (see lowerFunc).
		if g.retLLVM == types.I32 {
			if len(in.Uses) == 0 {
				b.NewRet(constant.NewInt(types.I32, 0))
				return nil
			}
			if g.loweringFn != nil {
				ty := g.findTypeOfSSA(g.loweringFn, in.Uses[0])
				if ty.Name == "none" {
					b.NewRet(constant.NewInt(types.I32, 0))
					return nil
				}
			}
			v := g.resolve(in.Uses[0])
			if v.Type().Equal(types.I64) {
				v = b.NewTrunc(v, types.I32)
			}
			b.NewRet(v)
			return nil
		}
		if len(in.Uses) == 0 || g.retLLVM.Equal(types.Void) {
			b.NewRet(nil)
			return nil
		}
		v := g.resolve(in.Uses[0])
		b.NewRet(v)
	case MIRSoaLoad:
		return g.lowerSoaLoad(in)
	case MIRSoaStore:
		return g.lowerSoaStore(in)
	case MIRHeapAlloc:
		if len(in.Uses) < 1 {
			return fmt.Errorf("heap_alloc: missing size")
		}
		mallocFn := g.ensureDecl("malloc", types.NewPointer(types.I8), ir.NewParam("sz", types.I64))
		sz := g.resolve(in.Uses[0])
		if !sz.Type().Equal(types.I64) {
			sz = b.NewSExt(sz, types.I64)
		}
		p := b.NewCall(mallocFn, sz)
		if in.Dst != 0 {
			g.vals[in.Dst] = p
		}
	case MIRPtrLoad:
		if len(in.Uses) < 1 {
			return fmt.Errorf("ptr_load: missing address")
		}
		p := g.resolve(in.Uses[0])
		pp := g.pointerCast(b, p, types.NewPointer(types.I32))
		v := b.NewLoad(types.I32, pp)
		if in.Dst != 0 {
			g.vals[in.Dst] = b.NewSExt(v, types.I64)
		}
	case MIRPtrStore:
		if len(in.Uses) < 2 {
			return fmt.Errorf("ptr_store: missing operands")
		}
		addr, val := g.resolve(in.Uses[0]), g.resolve(in.Uses[1])
		pp := g.pointerCast(b, addr, types.NewPointer(types.I32))
		v32 := val
		if val.Type().Equal(types.I64) {
			v32 = b.NewTrunc(val, types.I32)
		}
		b.NewStore(v32, pp)
	case MIRBorrow, MIRMutBorrow, MIRTimeTravelBorrow:
		// Ownership / timetravel metadata only; no LLVM.
		return nil
	case MIRChronoRead:
		// Compile-time temporal view: static analysis already proved safety; LLVM is identity.
		if len(in.Uses) < 1 {
			return fmt.Errorf("compiletime_temporal_view: missing operand")
		}
		v := g.resolve(in.Uses[0])
		if in.Dst != 0 {
			g.vals[in.Dst] = v
		}
	case MIRTemporalMove:
		if len(in.Uses) < 1 {
			return fmt.Errorf("temporal_move: missing value")
		}
		v := g.resolve(in.Uses[0])
		if in.Dst != 0 {
			g.vals[in.Dst] = v
		}
	case MIRParRegion:
		for _, ch := range in.Children {
			if err := g.lowerInst(&ch); err != nil {
				return err
			}
		}
	case MIRIfStrEq:
		return g.lowerIfStrEq(in)
	case MIRViewVec4List:
		return g.lowerViewVec4List(in)
	case MIRUnsafeRegion:
		for _, ch := range in.Children {
			if err := g.lowerInst(&ch); err != nil {
				return err
			}
		}
	case MIRSpacetimeRegion:
		// Compile-time region boundary only; zero extra LLVM instructions.
		for _, ch := range in.Children {
			if err := g.lowerInst(&ch); err != nil {
				return err
			}
		}
	default:
		g.warnf("unhandled MIR kind %v", in.Kind)
	}
	return nil
}

func (g *llvmGen) ensureDecl(name string, ret types.Type, params ...*ir.Param) *ir.Func {
	if f, ok := g.decls[name]; ok {
		return f
	}
	f := g.mod.NewFunc(name, ret, params...)
	g.decls[name] = f
	return f
}

func (g *llvmGen) coerceForStore(b *ir.Block, src value.Value, want types.Type) value.Value {
	if src.Type().Equal(want) {
		return src
	}
	if want.Equal(types.I64) && src.Type().Equal(types.I32) {
		return b.NewSExt(src, types.I64)
	}
	if want.Equal(types.I64) {
		if _, ok := src.Type().(*types.PointerType); ok {
			return b.NewPtrToInt(src, types.I64)
		}
	}
	if _, ok := want.(*types.PointerType); ok {
		return g.pointerCast(b, src, want)
	}
	if want.Equal(types.Double) && src.Type().Equal(types.I64) {
		return b.NewSIToFP(src, types.Double)
	}
	g.warnf("store: inserting bitcast %s -> %s", src.Type(), want)
	return b.NewBitCast(src, want)
}

// pointerCast is ptr→ptr bitcast, or i64→ptr inttoptr. LLVM/clang reject bitcast i64→ptr.
func (g *llvmGen) pointerCast(b *ir.Block, v value.Value, want types.Type) value.Value {
	if v.Type().Equal(want) {
		return v
	}
	if v.Type().Equal(types.I64) {
		return b.NewIntToPtr(v, want)
	}
	if _, ok := v.Type().(*types.PointerType); ok {
		return b.NewBitCast(v, want)
	}
	g.warnf("pointerCast: unexpected %s -> %s, using bitcast", v.Type(), want)
	return b.NewBitCast(v, want)
}

func (g *llvmGen) ptrToI64(b *ir.Block, v value.Value) value.Value {
	if v.Type().Equal(types.I64) {
		return v
	}
	if _, ok := v.Type().(*types.PointerType); ok {
		return b.NewPtrToInt(v, types.I64)
	}
	return b.NewBitCast(v, types.I64)
}

// lowerIfStrEq emits strcmp-based branch; Children = then, AltChildren = else.
func (g *llvmGen) lowerIfStrEq(in *MIRInst) error {
	if len(in.Uses) < 2 {
		return fmt.Errorf("if_str_eq: need two string operands")
	}
	b := g.block
	a := g.pointerCast(b, g.resolve(in.Uses[0]), types.NewPointer(types.I8))
	b2 := g.pointerCast(b, g.resolve(in.Uses[1]), types.NewPointer(types.I8))
	strcmpFn := g.ensureDecl("strcmp", types.I32,
		ir.NewParam("a", types.NewPointer(types.I8)),
		ir.NewParam("b", types.NewPointer(types.I8)))
	cmp := b.NewCall(strcmpFn, a, b2)
	z := constant.NewInt(types.I32, 0)
	cond := b.NewICmp(enum.IPredEQ, cmp, z)
	thenB := g.fn.NewBlock(g.freshName("if_then"))
	elseB := g.fn.NewBlock(g.freshName("if_else"))
	mergeB := g.fn.NewBlock(g.freshName("if_merge"))
	b.NewCondBr(cond, thenB, elseB)

	g.block = thenB
	for i := range in.Children {
		if err := g.lowerInst(&in.Children[i]); err != nil {
			return err
		}
	}
	if g.block.Term == nil {
		g.block.NewBr(mergeB)
	}

	g.block = elseB
	for i := range in.AltChildren {
		if err := g.lowerInst(&in.AltChildren[i]); err != nil {
			return err
		}
	}
	if g.block.Term == nil {
		g.block.NewBr(mergeB)
	}

	g.block = mergeB
	return nil
}

// lowerViewVec4List builds a stack { double* data; i64 len; i64 cap } and exposes it as i8* (roma4d_list_vec4_hdr).
func (g *llvmGen) lowerViewVec4List(in *MIRInst) error {
	if len(in.Uses) < 2 || in.Dst == 0 {
		return fmt.Errorf("view_vec4_list: need dst and two operands (ptr, len)")
	}
	b := g.block
	ptrRaw := g.pointerCast(b, g.resolve(in.Uses[0]), types.NewPointer(types.I8))
	nlen := g.resolve(in.Uses[1])
	if !nlen.Type().Equal(types.I64) {
		nlen = b.NewSExt(nlen, types.I64)
	}
	hdrTy := types.NewStruct(
		types.NewPointer(types.Double),
		types.I64,
		types.I64,
	)
	hdr := b.NewAlloca(hdrTy)
	hdr.SetName(g.freshName("vec4_list_hdr"))
	z := constant.NewInt(types.I32, 0)
	dataPtr := b.NewBitCast(ptrRaw, types.NewPointer(types.Double))
	gep0 := b.NewGetElementPtr(hdrTy, hdr, z, z)
	b.NewStore(dataPtr, gep0)
	i1 := constant.NewInt(types.I32, 1)
	gep1 := b.NewGetElementPtr(hdrTy, hdr, z, i1)
	b.NewStore(nlen, gep1)
	gep2 := b.NewGetElementPtr(hdrTy, hdr, z, constant.NewInt(types.I32, 2))
	b.NewStore(nlen, gep2)
	g.vals[in.Dst] = b.NewBitCast(hdr, types.NewPointer(types.I8))
	return nil
}

func (g *llvmGen) lowerGetItem(b *ir.Block, seq, idx value.Value) value.Value {
	// seq may be i8* (opaque list) — call runtime stub
	f := g.ensureDecl("roma4d_list_get_vec4", types.Void,
		ir.NewParam("lst", types.NewPointer(types.I8)),
		ir.NewParam("i", types.I64),
		ir.NewParam("out", types.NewPointer(types.Double)))
	out := b.NewAlloca(types.NewArray(4, types.Double))
	out.SetName(g.freshName("list_vec4_out"))
	ii := idx
	if !ii.Type().Equal(types.I64) {
		ii = b.NewSExt(idx, types.I64)
	}
	lst := g.pointerCast(b, seq, types.NewPointer(types.I8))
	b.NewCall(f, lst, ii, out)
	z0 := constant.NewInt(types.I32, 0)
	return b.NewGetElementPtr(types.NewArray(4, types.Double), out, z0, z0)
}

func (g *llvmGen) lowerSoaLoad(in *MIRInst) error {
	b := g.block
	if len(in.Uses) < 1 {
		return fmt.Errorf("soa_load: missing base")
	}
	base := g.resolve(in.Uses[0])
	cls, fidx, ok := g.lookupSoaField(in.ImmS)
	if !ok {
		g.warnf("soa_load: unknown field %q, using byte* load stub", in.ImmS)
		if in.Dst != 0 {
			g.vals[in.Dst] = base
		}
		return nil
	}
	st := g.classTys[cls]
	bt := types.NewPointer(st)
	ptr := g.pointerCast(b, base, bt)
	z := constant.NewInt(types.I32, 0)
	fi := constant.NewInt(types.I32, int64(fidx))
	gep := b.NewGetElementPtr(st, ptr, z, fi)
	// gep points to [4 x double]
	arrPtr := gep
	if in.Dst != 0 {
		g.vals[in.Dst] = arrPtr
	}
	return nil
}

func (g *llvmGen) lowerSoaStore(in *MIRInst) error {
	b := g.block
	if len(in.Uses) < 2 {
		return fmt.Errorf("soa_store: missing operands")
	}
	base, val := g.resolve(in.Uses[0]), g.resolve(in.Uses[1])
	cls, fidx, ok := g.lookupSoaField(in.ImmS)
	if !ok {
		return fmt.Errorf("soa_store: unknown field %q", in.ImmS)
	}
	st := g.classTys[cls]
	bt := types.NewPointer(st)
	ptr := g.pointerCast(b, base, bt)
	z := constant.NewInt(types.I32, 0)
	fi := constant.NewInt(types.I32, int64(fidx))
	gep := b.NewGetElementPtr(st, ptr, z, fi)
	// val is usually ptr to vec4 — memcpy 32 bytes
	mc := g.ensureDecl("memcpy", types.Void,
		ir.NewParam("dst", types.NewPointer(types.I8)),
		ir.NewParam("src", types.NewPointer(types.I8)),
		ir.NewParam("n", types.I64))
	dst8 := b.NewBitCast(gep, types.NewPointer(types.I8))
	src8 := b.NewBitCast(val, types.NewPointer(types.I8))
	b.NewCall(mc, dst8, src8, constant.NewInt(types.I64, 32))
	return nil
}

func (g *llvmGen) lookupSoaField(field string) (class string, idx int, ok bool) {
	for _, c := range g.mir.Classes {
		for i, f := range c.SoaFields {
			if f == field {
				return c.Name, i, true
			}
		}
	}
	return "", 0, false
}

func (g *llvmGen) lowerCall(in *MIRInst) error {
	b := g.block
	name := strings.Trim(in.Name, `"`)
	// C ABI: void identity_v4(double *out, const double *v). MIR only has the input vec4 pointer.
	if name == "identity_v4" && len(in.Uses) == 1 {
		vPtr := g.pointerCast(b, g.resolve(in.Uses[0]), types.NewPointer(types.Double))
		arrTy := types.NewArray(4, types.Double)
		outA := b.NewAlloca(arrTy)
		outA.SetName(g.freshName("id4_out"))
		z0 := constant.NewInt(types.I32, 0)
		outPtr := b.NewGetElementPtr(arrTy, outA, z0, z0)
		callee := g.declareForCallee("identity_v4", 2)
		b.NewCall(callee, outPtr, vPtr)
		vecI8 := g.pointerCast(b, outPtr, types.NewPointer(types.I8))
		if in.Dst != 0 {
			g.vals[in.Dst] = vecI8
		}
		return nil
	}
	callee := g.declareForCallee(name, len(in.Uses))
	var argVals []value.Value
	for i, u := range in.Uses {
		v := g.resolve(u)
		if i < len(callee.Sig.Params) {
			pt := callee.Sig.Params[i]
			v = g.coerceCallArg(b, v, pt)
		}
		argVals = append(argVals, v)
	}
	res := b.NewCall(callee, argVals...)
	if in.Dst != 0 {
		if callee.Sig.RetType.Equal(types.Void) {
			g.vals[in.Dst] = constant.NewInt(types.I64, 0)
		} else {
			g.vals[in.Dst] = res
		}
	}
	return nil
}

func (g *llvmGen) coerceCallArg(b *ir.Block, v value.Value, want types.Type) value.Value {
	if v.Type().Equal(want) {
		return v
	}
	if want.Equal(types.Double) && v.Type().Equal(types.I64) {
		return b.NewSIToFP(v, types.Double)
	}
	if want.Equal(types.NewPointer(types.I8)) {
		return g.pointerCast(b, v, want)
	}
	if want.Equal(types.I64) && v.Type().Equal(types.I32) {
		return b.NewSExt(v, types.I64)
	}
	return v
}

// declareForCallee creates or returns an extern declaration for MIR calls.
func (g *llvmGen) declareForCallee(name string, nArgs int) *ir.Func {
	if f, ok := g.decls[name]; ok {
		return f
	}
	var ret types.Type = types.I64
	var params []*ir.Param
	switch name {
	case "print", "puts":
		ret = types.I32
		params = []*ir.Param{ir.NewParam("s", types.NewPointer(types.I8))}
	case "malloc":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{ir.NewParam("n", types.I64)}
	case "mir_alloc":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{ir.NewParam("n", types.I64)}
	case "mir_ptr_load":
		ret = types.I64
		params = []*ir.Param{ir.NewParam("p", types.NewPointer(types.I8))}
	case "mir_ptr_store":
		ret = types.Void
		params = []*ir.Param{
			ir.NewParam("p", types.NewPointer(types.I8)),
			ir.NewParam("v", types.I64),
		}
	case "vec4":
		ret = types.NewPointer(types.I8)
		for i := 0; i < nArgs && len(params) < 8; i++ {
			params = append(params, ir.NewParam(fmt.Sprintf("a%d", i), types.Double))
		}
	case "rotor":
		ret = types.NewPointer(types.I8)
		if nArgs >= 1 {
			params = append(params, ir.NewParam("a0", types.Double))
		}
		if nArgs >= 2 {
			params = append(params, ir.NewParam("a1", types.NewPointer(types.I8)))
		}
	case "Particle", "multivector":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{}
	case "ollama_demo":
		ret = types.I32
		params = []*ir.Param{}
	case "quantum_server_demo":
		ret = types.I32
		params = []*ir.Param{}
	case "mir_mmap_gguf":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{ir.NewParam("path", types.NewPointer(types.I8))}
	case "mir_get_ollama_qwen_path":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{}
	case "mir_qwen_chat_loop":
		ret = types.I32
		params = []*ir.Param{}
	case "bump":
		ret = types.I32
		params = []*ir.Param{ir.NewParam("x", types.I32)}
	case "identity_v4":
		ret = types.Void
		params = []*ir.Param{
			ir.NewParam("out", types.NewPointer(types.Double)),
			ir.NewParam("v", types.NewPointer(types.Double)),
		}
	case "range", "len":
		ret = types.NewPointer(types.I8)
		params = []*ir.Param{ir.NewParam("a", types.I64)}
	default:
		ret = types.I64
		for i := 0; i < nArgs; i++ {
			params = append(params, ir.NewParam(fmt.Sprintf("a%d", i), types.I64))
		}
	}
	return g.ensureDecl(name, ret, params...)
}

func (g *llvmGen) resolve(id MIRValueID) value.Value {
	if id == 0 {
		return constant.NewInt(types.I64, 0)
	}
	if v, ok := g.vals[id]; ok {
		return v
	}
	g.warnf("undefined SSA %s, using 0", id.String())
	return constant.NewInt(types.I64, 0)
}

func mirTypeRefToLLVM(m MIRTypeRef, classes map[string]*types.StructType) types.Type {
	n := m.Name
	if i := strings.Index(n, "@"); i > 0 {
		n = n[:i]
	}
	if m.Linear && strings.HasPrefix(n, "linear[") {
		inner := strings.TrimSuffix(strings.TrimPrefix(n, "linear["), "]")
		return mirTypeRefToLLVM(MIRTypeRef{Name: inner}, classes)
	}
	switch {
	case n == "int" || n == "bool":
		return types.I64
	case n == "float":
		return types.Double
	case n == "rawptr" || strings.HasPrefix(n, "list["):
		return types.NewPointer(types.I8)
	case n == "vec4" || n == "rotor" || n == "multivector":
		return types.NewPointer(types.I8)
	case strings.HasPrefix(n, "class:"):
		cn := strings.TrimPrefix(n, "class:")
		if st, ok := classes[cn]; ok {
			return types.NewPointer(st)
		}
		return types.NewPointer(types.I8)
	case n == "none":
		return types.Void
	default:
		return types.I64
	}
}

func sanitizeIdent(s string) string {
	if s == "" {
		return "tmp"
	}
	// LLVM identifiers: avoid quotes from MIR
	s = strings.Trim(s, `"`)
	r := make([]rune, 0, len(s))
	for _, c := range s {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' {
			r = append(r, c)
		} else {
			r = append(r, '_')
		}
	}
	return string(r)
}
