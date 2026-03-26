package compiler

import "fmt"

// Scope is a lexical block: module, function, class, comprehension, par-region, etc.
type Scope struct {
	Parent   *Scope
	Children []*Scope

	Name string // for debugging: "module", "function main", "class Particle"

	// Symbols maps simple name -> symbol for this block only.
	Symbols map[string]*Symbol

	// Flags guide later passes (Ownership 2.0, par lowering).
	IsModule   bool
	IsFunction bool
	IsClass    bool
	IsParLoop  bool
}

// NewScope builds a child scope.
func NewScope(parent *Scope, name string) *Scope {
	s := &Scope{
		Parent:  parent,
		Name:    name,
		Symbols: make(map[string]*Symbol),
	}
	if parent != nil {
		parent.Children = append(parent.Children, s)
	}
	return s
}

// Define inserts a symbol; returns error if the name is already bound in this scope.
// The discard name "_" may be defined repeatedly (last binding wins).
func (s *Scope) Define(sym *Symbol) error {
	if sym == nil {
		return fmt.Errorf("internal error: nil symbol")
	}
	if sym.Name == DiscardName {
		sym.Discard = true
		s.Symbols[sym.Name] = sym
		return nil
	}
	if _, ok := s.Symbols[sym.Name]; ok {
		return fmt.Errorf("name %q is already defined in this scope", sym.Name)
	}
	s.Symbols[sym.Name] = sym
	return nil
}

// DefineAllowReplace overwrites a binding (loop variables, repl in Pass 4).
func (s *Scope) DefineAllowReplace(sym *Symbol) {
	if sym == nil {
		return
	}
	s.Symbols[sym.Name] = sym
}

// Lookup searches this scope then parents for a name.
func (s *Scope) Lookup(name string) *Symbol {
	for cur := s; cur != nil; cur = cur.Parent {
		if sym, ok := cur.Symbols[name]; ok {
			return sym
		}
	}
	return nil
}

// LookupLocal returns a symbol only if defined in this exact scope.
func (s *Scope) LookupLocal(name string) *Symbol {
	return s.Symbols[name]
}
