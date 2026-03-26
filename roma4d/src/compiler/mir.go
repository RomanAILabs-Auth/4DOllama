package compiler

import (
	"fmt"
	"strings"
)

// --- Mid-level IR (MIR) — SSA-oriented bridge to LLVM/CUDA (Pass 5) ---

// MIRValueID is an SSA value number (0 = none / void use).
type MIRValueID int

func (v MIRValueID) String() string {
	if v == 0 {
		return "_"
	}
	return fmt.Sprintf("%%%d", v)
}

// OwnershipMeta carries Pass 4 semantics on SSA defs (for codegen / validation).
type OwnershipMeta struct {
	IsLinear      bool
	Sendable      bool
	PythonTainted bool
	BorrowImm     bool
	BorrowMut     bool
	Notes         string
}

// MIRTypeRef is a compact type description in MIR (mirrors compiler.Type where needed).
type MIRTypeRef struct {
	Name    string // "int", "vec4", "rawptr", "linear[vec4]", "class:Particle", "vec4@t", etc.
	Linear  bool
	TimeTag string // Pass 7: non-empty when value is bound to a temporal slice (e.g. "t")
}

func mirTypeRefFromType(t Type) MIRTypeRef {
	if t == nil {
		return MIRTypeRef{Name: "any"}
	}
	if IsLinearType(t) {
		l := t.(Linear)
		return MIRTypeRef{Name: "linear[" + typeStr(l.Inner) + "]", Linear: true}
	}
	if st, ok := t.(SpacetimeType); ok {
		r := mirTypeRefFromType(st.Inner)
		tag := st.TimeTag
		if tag == "" {
			tag = "t"
		}
		r.Name = r.Name + "@" + tag
		r.TimeTag = tag
		return r
	}
	switch x := t.(type) {
	case TimeDim:
		return MIRTypeRef{Name: "time", TimeTag: "current"}
	case RawPtr:
		return MIRTypeRef{Name: "rawptr"}
	case Vec4:
		return MIRTypeRef{Name: "vec4"}
	case Rotor:
		return MIRTypeRef{Name: "rotor"}
	case Multivector:
		return MIRTypeRef{Name: "multivector"}
	case *Class:
		if x != nil {
			return MIRTypeRef{Name: "class:" + x.Name}
		}
	case *List:
		if x != nil && x.Elem != nil {
			return MIRTypeRef{Name: "list[" + typeStr(x.Elem) + "]"}
		}
	}
	return MIRTypeRef{Name: typeStr(t)}
}

// MIRInstKind classifies instructions.
type MIRInstKind int

const (
	MIRNop MIRInstKind = iota
	MIRConstInt
	MIRConstFloat
	MIRConstStr
	MIRLoadLocal // pseudo: read named local (pre-phi)
	MIRStoreLocal
	MIRCopy
	MIRBinOp
	MIRCall
	MIRGeomMul // vec4 * rotor etc.
	MIRReturn
	// SoA / ownership-visible memory
	MIRSoaLoad
	MIRSoaStore
	// Regions (nested bodies as separate MIR fragments)
	MIRParRegion
	MIRUnsafeRegion
	// Raw pointer / manual memory (systems)
	MIRHeapAlloc
	MIRPtrLoad
	MIRPtrStore
	// Borrow intrinsics (metadata + ordering for LLVM)
	MIRBorrow
	MIRMutBorrow
	// Pass 7 — spacetime (Option B)
	MIRTemporalMove     // value at time coordinate (@t)
	MIRSpacetimeRegion  // nested body with distinct temporal epoch
	MIRTimeTravelBorrow // borrow anchored for cross-moment observation (MIR metadata)
	MIRChronoRead       // Pass 8: compile-time temporal view; LLVM = identity (zero overhead)
	// String equality branch (strcmp); Children = then, AltChildren = else
	MIRIfStrEq
	// Zero-copy list[vec4] view over a raw pointer + logical length (vec4 count)
	MIRViewVec4List
	// Debug / placeholder
	MIRComment
)

// MIRInst is one SSA definition or side-effecting op.
type MIRInst struct {
	Kind  MIRInstKind
	Dst   MIRValueID
	Uses  []MIRValueID
	ImmI  int64
	ImmF  float64
	ImmS  string
	Name  string // local name, callee, field, etc.
	Ty    MIRTypeRef
	Own   OwnershipMeta
	Extra string
	// For ParRegion / UnsafeRegion: nested ops (not yet in CFG edge form)
	Children []MIRInst
	// For MIRIfStrEq: else branch
	AltChildren []MIRInst
}

// MIRBlock is a basic block (single successor v0; split in Pass 6).
type MIRBlock struct {
	Label string
	Insts []MIRInst
}

// MIRFunction is one lowered def.
type MIRFunction struct {
	Name       string
	QualModule string
	Params     []MIRParam
	Blocks     []MIRBlock
	Locals     []string // declared names for debug
}

// MIRParam is a formal parameter.
type MIRParam struct {
	Name string
	Ty   MIRTypeRef
}

// MIRClassDecl records SoA layout for lowering field accesses.
type MIRClassDecl struct {
	Name      string
	SoaFields []string
}

// MIRModule is one compilation unit after MIR lowering.
type MIRModule struct {
	SourcePath string
	Qual       string
	Classes    []MIRClassDecl
	Funcs      []*MIRFunction
}

// String returns a human-readable MIR dump (debug / tests).
func (m *MIRModule) String() string {
	if m == nil {
		return "<nil MIRModule>"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "module %q (%s)\n", m.Qual, m.SourcePath)
	for _, c := range m.Classes {
		fmt.Fprintf(&b, "class %s soa_fields=%v\n", c.Name, c.SoaFields)
	}
	for _, f := range m.Funcs {
		b.WriteString(f.String())
	}
	return b.String()
}

func (f *MIRFunction) String() string {
	if f == nil {
		return ""
	}
	var b strings.Builder
	params := make([]string, len(f.Params))
	for i, p := range f.Params {
		params[i] = fmt.Sprintf("%s:%s", p.Name, p.Ty.Name)
	}
	fmt.Fprintf(&b, "fn %s(%s) {\n", f.Name, strings.Join(params, ", "))
	for _, bl := range f.Blocks {
		fmt.Fprintf(&b, "  %s:\n", bl.Label)
		for _, in := range bl.Insts {
			b.WriteString("    ")
			b.WriteString(formatInst(&in))
			b.WriteByte('\n')
		}
	}
	b.WriteString("}\n")
	return b.String()
}

func formatInst(in *MIRInst) string {
	if in == nil {
		return ""
	}
	uses := make([]string, len(in.Uses))
	for i, u := range in.Uses {
		uses[i] = u.String()
	}
	ujoin := strings.Join(uses, ", ")
	own := ""
	if in.Own.IsLinear || in.Own.PythonTainted || in.Own.BorrowImm || in.Own.BorrowMut {
		own = fmt.Sprintf(" [own linear=%v taint=%v imm=%v mut=%v]",
			in.Own.IsLinear, in.Own.PythonTainted, in.Own.BorrowImm, in.Own.BorrowMut)
	}
	switch in.Kind {
	case MIRNop:
		return "nop"
	case MIRConstInt:
		if in.Dst != 0 {
			return fmt.Sprintf("%s = const i64 %d%s", in.Dst, in.ImmI, own)
		}
		return fmt.Sprintf("const i64 %d", in.ImmI)
	case MIRConstFloat:
		if in.Dst != 0 {
			return fmt.Sprintf("%s = const f64 %g%s", in.Dst, in.ImmF, own)
		}
		return fmt.Sprintf("const f64 %g", in.ImmF)
	case MIRConstStr:
		if in.Dst != 0 {
			return fmt.Sprintf("%s = const str %q%s", in.Dst, in.ImmS, own)
		}
		return fmt.Sprintf("const str %q", in.ImmS)
	case MIRLoadLocal:
		return fmt.Sprintf("%s = load_local %q ty=%s%s", in.Dst, in.Name, in.Ty.Name, own)
	case MIRStoreLocal:
		return fmt.Sprintf("store_local %q %s ty=%s%s", in.Name, ujoin, in.Ty.Name, own)
	case MIRCopy:
		return fmt.Sprintf("%s = copy %s%s", in.Dst, ujoin, own)
	case MIRBinOp:
		return fmt.Sprintf("%s = %s %s%s", in.Dst, in.Name, ujoin, own)
	case MIRCall:
		return fmt.Sprintf("%s = call %q(%s) -> %s%s", in.Dst, in.Name, ujoin, in.Ty.Name, own)
	case MIRGeomMul:
		return fmt.Sprintf("%s = geom_mul %s%s", in.Dst, ujoin, own)
	case MIRReturn:
		if len(in.Uses) == 0 {
			return "return" + own
		}
		return fmt.Sprintf("return %s%s", ujoin, own)
	case MIRSoaLoad:
		return fmt.Sprintf("%s = soa_load %s.%s ty=%s%s", in.Dst, ujoin, in.ImmS, in.Ty.Name, own)
	case MIRSoaStore:
		if len(in.Uses) >= 2 {
			return fmt.Sprintf("soa_store %s.%s %s%s", in.Uses[0].String(), in.ImmS, in.Uses[1].String(), own)
		}
		return fmt.Sprintf("soa_store ? %s%s", ujoin, own)
	case MIRParRegion:
		s := fmt.Sprintf("par_region loop_var=%q", in.Name)
		if in.Extra != "" {
			s += " extra=" + in.Extra
		}
		s += " {"
		for _, ch := range in.Children {
			s += " " + formatInst(&ch)
		}
		return s + " }"
	case MIRUnsafeRegion:
		s := "unsafe_region {"
		for _, ch := range in.Children {
			s += " " + formatInst(&ch)
		}
		return s + " }"
	case MIRHeapAlloc:
		return fmt.Sprintf("%s = heap_alloc bytes=%s ty=%s%s", in.Dst, ujoin, in.Ty.Name, own)
	case MIRPtrLoad:
		return fmt.Sprintf("%s = ptr_load %s%s", in.Dst, ujoin, own)
	case MIRPtrStore:
		if len(in.Uses) >= 2 {
			return fmt.Sprintf("ptr_store addr=%s value=%s%s", in.Uses[0].String(), in.Uses[1].String(), own)
		}
		return fmt.Sprintf("ptr_store %s%s", ujoin, own)
	case MIRBorrow:
		return fmt.Sprintf("borrow %s%s", ujoin, own)
	case MIRMutBorrow:
		return fmt.Sprintf("mutborrow %s%s", ujoin, own)
	case MIRTemporalMove:
		return fmt.Sprintf("%s = temporal_move [%s] @%s ty=%s ct_r=%d%s", in.Dst, ujoin, in.ImmS, in.Ty.Name, in.ImmI, own)
	case MIRSpacetimeRegion:
		s := fmt.Sprintf("spacetime_region ct_r=%d {", in.ImmI)
		for _, ch := range in.Children {
			s += " " + formatInst(&ch)
		}
		return s + " }"
	case MIRTimeTravelBorrow:
		return fmt.Sprintf("timetravel_borrow %s%s", ujoin, own)
	case MIRChronoRead:
		return fmt.Sprintf("%s = compiletime_temporal_view %s ct_r=%d%s", in.Dst, ujoin, in.ImmI, own)
	case MIRIfStrEq:
		s := "if_str_eq {"
		for _, ch := range in.Children {
			s += " " + formatInst(&ch)
		}
		s += " } else {"
		for _, ch := range in.AltChildren {
			s += " " + formatInst(&ch)
		}
		return s + " }"
	case MIRViewVec4List:
		return fmt.Sprintf("%s = view_vec4_list %s ty=%s", in.Dst, ujoin, in.Ty.Name)
	case MIRComment:
		return "// " + in.ImmS
	default:
		return fmt.Sprintf("?kind=%d %s", in.Kind, in.Extra)
	}
}

// HasGPUParSpacetime reports whether any `par` region is marked for GPU-candidate
// offload (nested under compile-time `spacetime:` in the source).
func (m *MIRModule) HasGPUParSpacetime() bool {
	if m == nil {
		return false
	}
	for _, f := range m.Funcs {
		if f == nil {
			continue
		}
		for _, bl := range f.Blocks {
			if mirInstListHasGPUParSpacetime(bl.Insts) {
				return true
			}
		}
	}
	return false
}

func mirInstListHasGPUParSpacetime(list []MIRInst) bool {
	for i := range list {
		in := &list[i]
		if in.Kind == MIRParRegion && in.Extra == "gpu_spacetime_par" {
			return true
		}
		if mirInstListHasGPUParSpacetime(in.Children) {
			return true
		}
		if mirInstListHasGPUParSpacetime(in.AltChildren) {
			return true
		}
	}
	return false
}
