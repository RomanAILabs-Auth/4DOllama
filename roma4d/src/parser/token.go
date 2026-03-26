package parser

// Token is one lexical unit with source position.
type Token struct {
	Kind TokenKind
	Lit  string
	Line int
	Col  int
}

// TokenKind enumerates Roma4D / Python 3.12 tokens plus r4d extensions.
type TokenKind int

const (
	ILLEGAL TokenKind = iota
	EOF
	NEWLINE
	INDENT
	DEDENT

	// Identifiers & literals
	IDENT
	INT
	FLOAT
	IMAG // complex literal ending in j
	STRING
	BYTES

	// Python + r4d keywords (tokenized as dedicated kinds for clarity)
	KwFalse
	KwNone
	KwTrue
	KwAnd
	KwAs
	KwAssert
	KwAsync
	KwAwait
	KwBreak
	KwClass
	KwContinue
	KwDef
	KwDel
	KwElif
	KwElse
	KwExcept
	KwFinally
	KwFor
	KwFrom
	KwGlobal
	KwIf
	KwImport
	KwIn
	KwIs
	KwLambda
	KwNonlocal
	KwNot
	KwOr
	KwPass
	KwRaise
	KwReturn
	KwTry
	KwWhile
	KwWith
	KwYield
	KwMatch
	KwCase
	KwType
	// Roma4D extensions
	KwPar
	KwSoa
	KwAos
	KwVec4
	KwRotor
	KwMultivector
	KwBorrow
	KwMutborrow
	KwUnsafe // systems: unsafe { } / unsafe: block
	KwTime   // current time coordinate `t` (spacetime axis)
	KwSpacetime

	// Delimiters / operators
	LPAREN   // (
	RPAREN   // )
	LBRACK   // [
	RBRACK   // ]
	LBRACE   // {
	RBRACE   // }
	COLON    // :
	COMMA    // ,
	SEMI     // ;
	DOT      // .
	ELLIPSIS // ...
	AT       // @
	ARROW    // ->
	EQEQ     // ==
	NOTEQ    // !=
	LTE      // <=
	GTE      // >=
	EQ       // =
	PLUS     // +
	MINUS    // -
	STAR     // *
	SLASH    // /
	DBLSLASH // //
	PERCENT  // %
	DBLSTAR  // **
	LT       // <
	GT       // >
	AMP      // &
	PIPE     // |
	CARET    // ^
	TILDE    // ~
	LSHIFT   // <<
	RSHIFT   // >>
	PLUSEQ   // +=
	MINUSEQ  // -=
	STAREQ   // *=
	SLASHEQ  // /=
	DBLSLASHEQ
	PERCENTEQ
	AMPEQ
	PIPEEQ
	CARETEQ
	LSHIFTEQ
	RSHIFTEQ
	DBLSTAREQ
	ATEQ     // @=
	LPAREQ   // ( not used — placeholder avoided
	EXCL     // !
	COLONEQ  // :=
)

func (k TokenKind) IsKeyword() bool {
	return k >= KwFalse && k <= KwSpacetime
}

// KeywordTable maps spellings to token kinds for Python 3.12 + r4d.
var KeywordTable = map[string]TokenKind{
	"False":       KwFalse,
	"None":        KwNone,
	"True":        KwTrue,
	"and":         KwAnd,
	"as":          KwAs,
	"assert":      KwAssert,
	"async":       KwAsync,
	"await":       KwAwait,
	"break":       KwBreak,
	"class":       KwClass,
	"continue":    KwContinue,
	"def":         KwDef,
	"del":         KwDel,
	"elif":        KwElif,
	"else":        KwElse,
	"except":      KwExcept,
	"finally":     KwFinally,
	"for":         KwFor,
	"from":        KwFrom,
	"global":      KwGlobal,
	"if":          KwIf,
	"import":      KwImport,
	"in":          KwIn,
	"is":          KwIs,
	"lambda":      KwLambda,
	"nonlocal":    KwNonlocal,
	"not":         KwNot,
	"or":          KwOr,
	"pass":        KwPass,
	"raise":       KwRaise,
	"return":      KwReturn,
	"try":         KwTry,
	"while":       KwWhile,
	"with":        KwWith,
	"yield":       KwYield,
	"match":       KwMatch,
	"case":        KwCase,
	"type":        KwType,
	"par":         KwPar,
	"soa":         KwSoa,
	"aos":         KwAos,
	"vec4":        KwVec4,
	"rotor":       KwRotor,
	"multivector": KwMultivector,
	"borrow":      KwBorrow,
	"mutborrow":   KwMutborrow,
	"unsafe":      KwUnsafe,
	"t":           KwTime,
	"spacetime":   KwSpacetime,
}
