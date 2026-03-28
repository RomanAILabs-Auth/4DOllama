package parser

// Module is a parsed file (exec module body).
type Module struct {
	Filename string
	Body     []Stmt
	// Qual is the logical module id after package resolution (e.g. "__main__", "mylib.utils").
	Qual string
}

// Stmt is implemented by all statement nodes.
type Stmt interface {
	stmt()
}

// Expr is implemented by all expression nodes.
type Expr interface {
	expr()
}

// --- Statements ---

// FunctionDef is `def name(args) [-> ret]: suite`.
type FunctionDef struct {
	Async      bool
	Name       string
	Args       *Arguments
	Returns    Expr // optional annotation
	Body       []Stmt
	Decorators []Expr
	// Sym is the function symbol after name resolution; Typ is its function type after checking.
	Sym any
	Typ any
}

func (*FunctionDef) stmt() {}

// ClassDef is `class Name[(bases)]: suite`.
type ClassDef struct {
	Name       string
	Bases      []Expr
	Keywords   []*Keyword // metaclass=..., etc.
	Body       []Stmt
	Decorators []Expr
	Sym        any
	Typ        any
}

func (*ClassDef) stmt() {}

// ReturnStmt is `return` or `return a, b`.
type ReturnStmt struct {
	Values []Expr
}

func (*ReturnStmt) stmt() {}

// AssignStmt is `targets op value` (op is EQ or augmented assign kind).
type AssignStmt struct {
	Targets []Expr
	Op      TokenKind // EQ, PLUSEQ, ...
	Value   Expr
}

func (*AssignStmt) stmt() {}

// AnnAssign is `target: annotation [= value]` with optional SoA/AoS layout prefix.
type AnnAssign struct {
	Layout     *TokenKind // KwSoa or KwAos when present
	Target     Expr
	Annotation Expr
	Value      Expr // nil if no `=`
	Sym        any
	Typ        any
}

func (*AnnAssign) stmt() {}

// ExprStmt is a bare expression (often call).
type ExprStmt struct {
	Value Expr
}

func (*ExprStmt) stmt() {}

// IfStmt is `if` / `elif` / `else`.
type IfStmt struct {
	Test  Expr
	Body  []Stmt
	Elifs []ElifClause
	Else  []Stmt
}

// ElifClause is one elif branch.
type ElifClause struct {
	Test Expr
	Body []Stmt
}

func (*IfStmt) stmt() {}

// WhileStmt is `while test: suite`.
type WhileStmt struct {
	Test Expr
	Body []Stmt
	Else []Stmt
}

func (*WhileStmt) stmt() {}

// ForStmt is `for target in iter: suite`.
type ForStmt struct {
	Async bool
	Target Expr
	Iter   Expr
	Body   []Stmt
	Else   []Stmt
}

func (*ForStmt) stmt() {}

// ParForStmt is `par for target in iter: suite` (Roma4D extension).
type ParForStmt struct {
	Async  bool
	Target Expr
	Iter   Expr
	Body   []Stmt
	Else   []Stmt
}

func (*ParForStmt) stmt() {}

// PassStmt, BreakStmt, ContinueStmt are simple.
type PassStmt struct{}
type BreakStmt struct{}
type ContinueStmt struct{}

func (*PassStmt) stmt()    {}
func (*BreakStmt) stmt()   {}
func (*ContinueStmt) stmt() {}

// DeleteStmt is `del targets`.
type DeleteStmt struct {
	Targets []Expr
}

func (*DeleteStmt) stmt() {}

// AssertStmt is `assert test [, msg]`.
type AssertStmt struct {
	Test Expr
	Msg  Expr
}

func (*AssertStmt) stmt() {}

// RaiseStmt is `raise` with optional parts.
type RaiseStmt struct {
	Exc   Expr
	Cause Expr
}

func (*RaiseStmt) stmt() {}

// GlobalStmt / NonlocalStmt.
type GlobalStmt struct {
	Names []string
}

func (*GlobalStmt) stmt() {}

type NonlocalStmt struct {
	Names []string
}

func (*NonlocalStmt) stmt() {}

// ImportStmt is `import a, b as c`.
type ImportStmt struct {
	Names []Alias
}

func (*ImportStmt) stmt() {}

// ImportFrom is `from x import y`.
type ImportFrom struct {
	Module   string
	Names    []Alias
	Star     bool
	Relative int // leading dots count
}

func (*ImportFrom) stmt() {}

// TryStmt is try / except / else / finally.
type TryStmt struct {
	Body      []Stmt
	Handlers  []ExceptHandler
	Else      []Stmt
	Finally   []Stmt
}

func (*TryStmt) stmt() {}

// ExceptHandler is `except [type as name]: suite`.
type ExceptHandler struct {
	Type Expr
	Name string
	Body []Stmt
}

// WithStmt is `with items: suite`.
type WithStmt struct {
	Items []WithItem
	Body  []Stmt
}

func (*WithStmt) stmt() {}

// UnsafeStmt is `unsafe:` suite — systems region (manual memory / raw ops); MIR lowers to unsafe region.
type UnsafeStmt struct {
	Body []Stmt
}

func (*UnsafeStmt) stmt() {}

// SpacetimeStmt is `spacetime:` suite — opens a new temporal epoch for borrow / MIR regions (Pass 7).
type SpacetimeStmt struct {
	Body []Stmt
}

func (*SpacetimeStmt) stmt() {}

// TimeCoord is the keyword `t` — the current evaluation's time coordinate (Pass 7).
type TimeCoord struct {
	Line int
	Col  int
	Typ  any // set by type checker (compiler.Type)
}

func (*TimeCoord) expr() {}

// WithItem is `expr [as target]`.
type WithItem struct {
	Context Expr
	Target  Expr
}

// MatchStmt for pattern matching (subset).
type MatchStmt struct {
	Subject Expr
	Cases   []MatchCase
}

func (*MatchStmt) stmt() {}

// MatchCase is `case pattern [: guard]: suite`.
type MatchCase struct {
	Pattern Expr
	Guard   Expr
	Body    []Stmt
}

// TypeAliasStmt is `type Name = value` (Python 3.12).
type TypeAliasStmt struct {
	Name  string
	Value Expr
}

func (*TypeAliasStmt) stmt() {}

// --- Expressions ---

// Name is an identifier reference.
type Name struct {
	Id  string
	Line int
	Col  int
	Sym any // *compiler.Symbol after resolution (see compiler package)
	Typ any // compiler.Type after inference
	// OwnMeta is reserved for mid-level IR / ownership notes (Pass 5).
	OwnMeta any
}

func (*Name) expr() {}

// Constant wraps a literal (string from lexer for numbers).
type Constant struct {
	Kind TokenKind // INT, FLOAT, IMAG, STRING
	Text string
	Typ  any // inferred primitive type
}

func (*Constant) expr() {}

// Tuple, List, Set, Dict literals.
type TupleExpr struct {
	Elts      []Expr
	Unpacking bool // trailing comma single element
}

func (*TupleExpr) expr() {}

type ListExpr struct {
	Elts []Expr
}

func (*ListExpr) expr() {}

type SetExpr struct {
	Elts []Expr
}

func (*SetExpr) expr() {}

type DictExpr struct {
	Keys   []Expr
	Values []Expr
}

func (*DictExpr) expr() {}

// BinOp is binary operator. For multivectors, STAR/CARET/PIPE become geometric product / outer / inner after typing.
type BinOp struct {
	Op    TokenKind
	Left  Expr
	Right Expr
	Typ   any // result type (geometric vs Python semantics)
}

func (*BinOp) expr() {}

// BoolOp is `and` / `or` chain.
type BoolOp struct {
	Op     TokenKind
	Values []Expr
}

func (*BoolOp) expr() {}

// UnaryOp is unary + - ~ not.
type UnaryOp struct {
	Op   TokenKind
	Expr Expr
}

func (*UnaryOp) expr() {}

// Compare is chained comparison `a < b < c`.
type Compare struct {
	Left        Expr
	Ops         []TokenKind
	Comparators []Expr
}

func (*Compare) expr() {}

// Keyword is `name=value` in call or class header.
type Keyword struct {
	Name string // empty for **kwargs spread (future)
	Val  Expr
}

// Call is `func(args)`.
type Call struct {
	Func     Expr
	Args     []Expr
	Keywords []*Keyword
	Sym      any // callee symbol when applicable
	Typ      any // result type
}

func (*Call) expr() {}

// IfExp is `a if cond else b`.
type IfExp struct {
	Test Expr
	Body Expr
	Orelse Expr
}

func (*IfExp) expr() {}

// Attribute is `value.attr`.
type Attribute struct {
	Value Expr
	Attr  string
	Sym   any // field or method symbol
	Typ   any
}

func (*Attribute) expr() {}

// Subscript is `value[key]` or slice.
type Subscript struct {
	Value Expr
	Slice Expr
	Typ   any
}

func (*Subscript) expr() {}

// Slice expression (low:high:step).
type SliceExpr struct {
	Lower Expr
	Upper Expr
	Step  Expr
}

func (*SliceExpr) expr() {}

// Starred is `*x` in call/list/tuple.
type Starred struct {
	Value Expr
}

func (*Starred) expr() {}

// Lambda is `lambda args: expr`.
type Lambda struct {
	Args *Arguments
	Body Expr
}

func (*Lambda) expr() {}

// Await, Yield, YieldFrom (placeholders).
type AwaitExpr struct {
	Value Expr
}

func (*AwaitExpr) expr() {}

// ListComp, SetComp, DictComp, GeneratorExp.
type ListComp struct {
	Elt   Expr
	Comps []Comprehension
}

func (*ListComp) expr() {}

type SetComp struct {
	Elt   Expr
	Comps []Comprehension
}

func (*SetComp) expr() {}

type DictComp struct {
	Key   Expr
	Value Expr
	Comps []Comprehension
}

func (*DictComp) expr() {}

type GeneratorExp struct {
	Elt   Expr
	Comps []Comprehension
}

func (*GeneratorExp) expr() {}

// Comprehension is `for x in it [if cond]`.
type Comprehension struct {
	Async  bool
	Target Expr
	Iter   Expr
	Ifs    []Expr
}

// Arguments for def/call/lambda.
type Arguments struct {
	Args     []*Arg
	Vararg   string // *name or * (posonly end)
	Kwonly   []*Arg
	Kwarg    string // **name
}

// Arg is one formal parameter with optional annotation/default.
type Arg struct {
	Name    string
	Annot   Expr
	Default Expr
}

// Alias for import.
type Alias struct {
	Name string
	As   string
}
