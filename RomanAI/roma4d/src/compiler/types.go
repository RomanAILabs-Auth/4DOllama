package compiler

import "fmt"

// Type is a Roma4D / Python v0 type. Geometric types mirror the semantics of the
// golden reference in src/core/4d/ (ga4 package: Vec4, Rotor, Multivector, Cl(4,0)).
type Type interface {
	TypeString() string
}

// --- Primitives (Python) ---

type PrimKind int

const (
	PrimAny PrimKind = iota
	PrimNone
	PrimBool
	PrimInt
	PrimFloat
	PrimComplex
	PrimStr
	PrimBytes
)

// Primitive is a Python builtin scalar class.
type Primitive struct {
	Kind PrimKind
}

func (p *Primitive) TypeString() string {
	switch p.Kind {
	case PrimNone:
		return "None"
	case PrimBool:
		return "bool"
	case PrimInt:
		return "int"
	case PrimFloat:
		return "float"
	case PrimComplex:
		return "complex"
	case PrimStr:
		return "str"
	case PrimBytes:
		return "bytes"
	default:
		return "Any"
	}
}

// --- Native 4D (Cl(4,0)) — built-ins, no import required ---

// Vec4 is a Euclidean 4-vector (grade-1 embedding); maps to ga4.Vec4.
type Vec4 struct{}

func (Vec4) TypeString() string { return "vec4" }

// Rotor is an even subalgebra rotor; maps to ga4.Rotor.
type Rotor struct{}

func (Rotor) TypeString() string { return "rotor" }

// Multivector is a full Cl(4,0) 16-component element; maps to ga4.Multivector.
type Multivector struct{}

func (Multivector) TypeString() string { return "multivector" }

// TimeDim is the built-in scalar time axis (keyword `t`); Pass 7 spacetime foundation.
type TimeDim struct{}

func (TimeDim) TypeString() string { return "time" }

// SpacetimeType binds Inner to an explicit time slice (e.g. `expr @ t`). Pass 7 Option B.
// RegionID is a compile-time static region index (0 = root timeline); it exists only for
// the checker/MIR and is erased before code generation — zero runtime cost.
type SpacetimeType struct {
	Inner    Type
	TimeTag  string // "t" for current-coordinate binding
	RegionID int    // compile-time spacetime region; never emitted to the binary
}

func (s SpacetimeType) TypeString() string {
	if s.Inner == nil {
		return "spacetime[?]"
	}
	tag := s.TimeTag
	if tag == "" {
		tag = "t"
	}
	base := s.Inner.TypeString() + "@" + tag
	if s.RegionID != 0 {
		return base + fmt.Sprintf("[r%d]", s.RegionID)
	}
	return base
}

// TemporalRegionMeta is a compile-time-only description of a `spacetime:` boundary.
// It never materializes as a runtime value or LLVM type; static analysis uses it for
// causal ordering and borrow scoping without emitting code.
type TemporalRegionMeta struct {
	CompileTimeID int
}

func (TemporalRegionMeta) TypeString() string { return "temporal_region(ct)" }

// --- Collections ---

// List is a homogeneous list (v0).
type List struct {
	Elem Type
}

func (l *List) TypeString() string {
	if l == nil || l.Elem == nil {
		return "list"
	}
	return "list[" + l.Elem.TypeString() + "]"
}

// Tuple holds fixed element types (v0: optional arity).
type Tuple struct {
	Elts []Type
}

func (t *Tuple) TypeString() string {
	if t == nil || len(t.Elts) == 0 {
		return "tuple"
	}
	s := "tuple["
	for i, e := range t.Elts {
		if i > 0 {
			s += ", "
		}
		if e == nil {
			s += "Any"
		} else {
			s += e.TypeString()
		}
	}
	return s + "]"
}

// Callable is a function or type constructor.
type Callable struct {
	Params     []Type // nil = unchecked varargs
	Variadic   bool
	KwVariadic bool
	Result     Type
	// IsCtor marks native/type constructors (vec4, rotor, …).
	IsCtor bool
}

func (c *Callable) TypeString() string {
	if c == nil {
		return "callable"
	}
	if c.Result != nil {
		return "Callable[..., " + c.Result.TypeString() + "]"
	}
	return "Callable[..., Any]"
}

// Class is a user-defined class type (fields include SoA/AoS layout metadata).
type Class struct {
	Name   string
	Fields map[string]*Field
	// BaseNames holds unresolved base identifiers (Pass 3 v0).
	BaseNames []string
}

// Field is one member with optional memory layout (Ownership 2.0 / SoA hooks).
type Field struct {
	Name    string
	Type    Type
	Layout  LayoutKind
	Linear  bool // reserved for Pass 4 linear / borrow checking
	Mutable bool // reserved for Pass 4
}

// LayoutKind mirrors soa / aos annotations on class bodies.
type LayoutKind int

const (
	LayoutNone LayoutKind = iota
	LayoutSoa
	LayoutAos
)

func (c *Class) TypeString() string {
	if c == nil {
		return "class"
	}
	return "class " + c.Name
}

// ModuleType is a loaded submodule namespace.
type ModuleType struct {
	Qual   string
	Exports map[string]Type
}

func (m *ModuleType) TypeString() string {
	if m == nil {
		return "module"
	}
	return "module " + m.Qual
}

// Linear wraps a move-only value (typically a read from an `soa` column).
type Linear struct {
	Inner Type
}

func (l Linear) TypeString() string {
	if l.Inner == nil {
		return "linear"
	}
	return "linear[" + l.Inner.TypeString() + "]"
}

// BorrowedRef is an immutable shared borrow; safe to capture in `par for`.
type BorrowedRef struct {
	Inner Type
}

func (b *BorrowedRef) TypeString() string {
	if b == nil || b.Inner == nil {
		return "Borrowed"
	}
	return "Borrowed[" + b.Inner.TypeString() + "]"
}

// MutBorrowRef is an exclusive mutable borrow; not Sendable for `par`.
type MutBorrowRef struct {
	Inner Type
}

func (m *MutBorrowRef) TypeString() string {
	if m == nil || m.Inner == nil {
		return "MutBorrowed"
	}
	return "MutBorrowed[" + m.Inner.TypeString() + "]"
}

// StripLinear unwraps one Linear layer (used for geometric typing).
func StripLinear(t Type) Type {
	if l, ok := t.(Linear); ok {
		return l.Inner
	}
	return t
}

// StripSpacetime unwraps one SpacetimeType layer (GA / numeric checks use the spatial core type).
func StripSpacetime(t Type) Type {
	if st, ok := t.(SpacetimeType); ok {
		return st.Inner
	}
	return t
}

// IsLinearType reports whether t is a move-only linear value.
func IsLinearType(t Type) bool {
	_, ok := t.(Linear)
	return ok
}

// TypeIsSendable reports whether a value of type t may be captured by `par for`.
func TypeIsSendable(t Type) bool {
	if t == nil {
		return true
	}
	switch x := t.(type) {
	case *BorrowedRef:
		return true
	case *MutBorrowRef:
		return false
	case SpacetimeType:
		return TypeIsSendable(x.Inner)
	case Linear:
		return false
	case *Primitive:
		return true
	case Vec4, Rotor, Multivector:
		return true
	case *Class:
		return !classHasSoaFields(x)
	case *List, *Tuple, *Callable, *ModuleType:
		return true
	case RawPtr:
		return false
	}
	return true
}

func classHasSoaFields(c *Class) bool {
	if c == nil {
		return false
	}
	for _, f := range c.Fields {
		if f != nil && f.Layout == LayoutSoa {
			return true
		}
	}
	return false
}

// RawPtr is an opaque raw pointer (systems / MIR heap_alloc); not Sendable by default.
type RawPtr struct{}

func (RawPtr) TypeString() string { return "rawptr" }

// Union is a loose union for error recovery.
type Union struct {
	Options []Type
}

func (u *Union) TypeString() string { return "Union" }

// Predeclared singletons for built-in geometric types.
var (
	TypVec4        Type = Vec4{}
	TypRotor       Type = Rotor{}
	TypMultivector Type = Multivector{}
	TypAny         Type = &Primitive{Kind: PrimAny}
	TypNone        Type = &Primitive{Kind: PrimNone}
	TypBool        Type = &Primitive{Kind: PrimBool}
	TypInt         Type = &Primitive{Kind: PrimInt}
	TypFloat       Type = &Primitive{Kind: PrimFloat}
	TypStr         Type = &Primitive{Kind: PrimStr}
	TypRawPtr      Type = RawPtr{}
	TypTime        Type = TimeDim{}
)

func isNumeric(t Type) bool {
	switch t.(type) {
	case *Primitive:
		p := t.(*Primitive)
		return p.Kind == PrimInt || p.Kind == PrimFloat || p.Kind == PrimComplex
	default:
		return false
	}
}

func isGA(t Type) bool {
	t = StripLinear(t)
	t = StripSpacetime(t)
	switch t.(type) {
	case Vec4, Rotor, Multivector:
		return true
	default:
		return false
	}
}

// PromoteNumeric picks float if either side is float.
func PromoteNumeric(a, b Type) Type {
	ai, aIs := a.(*Primitive)
	bi, bIs := b.(*Primitive)
	if !aIs || !bIs {
		return TypAny
	}
	if ai.Kind == PrimFloat || bi.Kind == PrimFloat {
		return TypFloat
	}
	if ai.Kind == PrimComplex || bi.Kind == PrimComplex {
		return &Primitive{Kind: PrimComplex}
	}
	return TypInt
}
