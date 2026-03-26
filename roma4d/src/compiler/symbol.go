package compiler

// SymbolKind classifies definitions for resolution and (later) ownership.
type SymbolKind int

const (
	SymVar SymbolKind = iota
	SymParam
	SymFunc
	SymClass
	SymField
	SymImport
	SymBuiltin
	SymModule
	SymTypeAlias
)

// DiscardName is the anonymous binding that may be assigned repeatedly in one scope.
const DiscardName = "_"

// Symbol is a resolved name in a scope. Pass 4 will attach borrow / sendability here.
type Symbol struct {
	Name string
	Kind SymbolKind
	Type Type
	// Decl is the defining AST node (parser type as interface{} to avoid cycles in docs).
	Decl interface{}

	// ImportPath is set for imported modules (logical path).
	ImportPath string
	// Sendable is true if the value may be captured by `par for` (see Ownership 2.0).
	Sendable bool
	// LinearHint marks declarations tied to SoA / linear resources.
	LinearHint bool
	// DefScope is the lexical scope where this symbol was defined (nil for builtins).
	DefScope *Scope
	// TaintedPython is set when the value is passed into Python interop (e.g. print).
	TaintedPython bool
	// Discard is true for the special "_" binding (redefinition allowed in the same scope).
	Discard bool
}

func (s *Symbol) String() string {
	if s == nil {
		return "<nil>"
	}
	return s.Name
}
