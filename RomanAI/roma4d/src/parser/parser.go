package parser

import (
	"fmt"
	"os"
	"strconv"
)

// Parser turns tokens into an AST.
type Parser struct {
	path string
	lx   *Lexer
	tok  Token
}

// ParseFile reads filename and parses it into a Module.
func ParseFile(filename string) (*Module, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return Parse(filename, string(b))
}

// Parse parses source text into a Module (exec form).
func Parse(filename, src string) (*Module, error) {
	src = stripLeadingUTF8BOM(src)
	p := &Parser{
		path: filename,
		lx:   NewLexer(filename, src),
	}
	p.next()
	m := &Module{Filename: filename}
	for p.tok.Kind != EOF {
		if p.tok.Kind == NEWLINE {
			p.next()
			continue
		}
		stmts, err := p.parseStmtList(false)
		if err != nil {
			return nil, err
		}
		m.Body = append(m.Body, stmts...)
	}
	if p.tok.Kind != EOF {
		return nil, p.errf("expected EOF, got %s", p.tok.Lit)
	}
	return m, nil
}

func (p *Parser) next() { p.tok = p.lx.Next() }

func (p *Parser) errf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s:%d:%d: %s", p.path, p.tok.Line, p.tok.Col, msg)
}

func (p *Parser) want(kind TokenKind) error {
	if p.tok.Kind != kind {
		return p.errf("expected %s, got %s", tokString(kind), tokString(p.tok.Kind))
	}
	p.next()
	return nil
}

func tokString(k TokenKind) string {
	if k == IDENT {
		return "identifier"
	}
	if k == NEWLINE {
		return "newline"
	}
	if k == INDENT {
		return "INDENT"
	}
	if k == DEDENT {
		return "DEDENT"
	}
	if k == EOF {
		return "end of file"
	}
	if k.IsKeyword() {
		return "keyword " + k.String()
	}
	return strconv.Itoa(int(k))
}

// String returns a readable name for keyword TokenKinds (for errors).
func (k TokenKind) String() string {
	for s, kk := range KeywordTable {
		if kk == k {
			return s
		}
	}
	switch k {
	case EOF:
		return "EOF"
	case NEWLINE:
		return "NEWLINE"
	case INDENT:
		return "INDENT"
	case DEDENT:
		return "DEDENT"
	case IDENT:
		return "IDENT"
	case STRING:
		return "STRING"
	case INT:
		return "INT"
	case FLOAT:
		return "FLOAT"
	case IMAG:
		return "IMAG"
	default:
		return "token"
	}
}

// parseStmtList parses statements until EOF (module) or DEDENT (block).
func (p *Parser) parseStmtList(stopAtDed bool) ([]Stmt, error) {
	var out []Stmt
	for p.tok.Kind != EOF && (!stopAtDed || p.tok.Kind != DEDENT) {
		if p.tok.Kind == NEWLINE {
			p.next()
			continue
		}
		st, err := p.parseStmt()
		if err != nil {
			return nil, err
		}
		out = append(out, st...)
	}
	if stopAtDed && p.tok.Kind == DEDENT {
		p.next()
	}
	return out, nil
}

// parseStmt returns one or more statements (semicolon-separated simple stmts).
func (p *Parser) parseStmt() ([]Stmt, error) {
	if p.tok.Kind == AT {
		return p.parseDecorated()
	}
	if p.tok.Kind == KwSoa || p.tok.Kind == KwAos {
		return p.parseLayoutFieldStmt()
	}
	if p.isCompoundStart() {
		s, err := p.parseCompoundStmt()
		if err != nil {
			return nil, err
		}
		if err := p.expectEndOfSimpleLine(); err != nil {
			return nil, err
		}
		return []Stmt{s}, nil
	}
	return p.parseSimpleStmts()
}

func (p *Parser) expectEndOfSimpleLine() error {
	if p.tok.Kind == NEWLINE {
		p.next()
		return nil
	}
	if p.tok.Kind == EOF || p.tok.Kind == DEDENT {
		return nil
	}
	if p.tok.Kind == SEMI {
		return nil
	}
	// After DEDENT, the lexer goes straight to the next statement's first token (no extra NEWLINE).
	if p.isStmtStart() {
		return nil
	}
	return p.errf("expected newline after compound statement, got %s", p.tok.Lit)
}

func (p *Parser) isStmtStart() bool {
	if p.isCompoundStart() {
		return true
	}
	switch p.tok.Kind {
	case AT, KwPass, KwBreak, KwContinue, KwReturn, KwRaise, KwDel, KwAssert,
		KwGlobal, KwNonlocal, KwImport, KwFrom, IDENT, STRING, INT, FLOAT, IMAG,
		TILDE, PLUS, MINUS, LPAREN, LBRACK, LBRACE, KwSoa, KwAos, KwNot, KwLambda, KwAwait,
		KwBorrow, KwMutborrow, KwTime:
		return true
	default:
		return false
	}
}

func (p *Parser) isCompoundStart() bool {
	switch p.tok.Kind {
	case KwDef, KwClass, KwIf, KwWhile, KwFor, KwTry, KwWith, KwMatch, KwType, KwAsync, KwUnsafe, KwSpacetime:
		return true
	case KwPar:
		return true
	default:
		return false
	}
}

func (p *Parser) parseSimpleStmts() ([]Stmt, error) {
	var out []Stmt
	for {
		s, err := p.parseSmallStmt()
		if err != nil {
			return nil, err
		}
		out = append(out, s)
		if p.tok.Kind == SEMI {
			p.next()
			if p.tok.Kind == NEWLINE || p.tok.Kind == EOF || p.tok.Kind == DEDENT {
				break
			}
			continue
		}
		break
	}
	if p.tok.Kind == NEWLINE {
		p.next()
	} else if p.tok.Kind != EOF && p.tok.Kind != DEDENT {
		return nil, p.errf("expected newline or ;, got %s", p.tok.Lit)
	}
	return out, nil
}

func (p *Parser) parseSmallStmt() (Stmt, error) {
	switch p.tok.Kind {
	case KwPass:
		p.next()
		return &PassStmt{}, nil
	case KwBreak:
		p.next()
		return &BreakStmt{}, nil
	case KwContinue:
		p.next()
		return &ContinueStmt{}, nil
	case KwReturn:
		p.next()
		return p.parseReturnStmt()
	case KwRaise:
		p.next()
		return p.parseRaiseStmt()
	case KwDel:
		p.next()
		return p.parseDelStmt()
	case KwAssert:
		p.next()
		return p.parseAssertStmt()
	case KwGlobal:
		p.next()
		return p.parseGlobalStmt()
	case KwNonlocal:
		p.next()
		return p.parseNonlocalStmt()
	case KwImport:
		p.next()
		return p.parseImportStmt()
	case KwFrom:
		return p.parseImportFromStmt()
	default:
		return p.parseAssignOrExprStmt()
	}
}

func (p *Parser) parseReturnStmt() (Stmt, error) {
	if p.tok.Kind == NEWLINE || p.tok.Kind == SEMI || p.tok.Kind == EOF || p.tok.Kind == DEDENT {
		return &ReturnStmt{}, nil
	}
	es, err := p.parseExprList()
	if err != nil {
		return nil, err
	}
	return &ReturnStmt{Values: es}, nil
}

func (p *Parser) parseRaiseStmt() (Stmt, error) {
	if p.tok.Kind == NEWLINE || p.tok.Kind == SEMI || p.tok.Kind == EOF || p.tok.Kind == DEDENT {
		return &RaiseStmt{}, nil
	}
	exc, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	rs := &RaiseStmt{Exc: exc}
	if p.tok.Kind == KwFrom {
		p.next()
		c, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		rs.Cause = c
	}
	return rs, nil
}

func (p *Parser) parseDelStmt() (Stmt, error) {
	ts, err := p.parseExprList()
	if err != nil {
		return nil, err
	}
	return &DeleteStmt{Targets: ts}, nil
}

func (p *Parser) parseAssertStmt() (Stmt, error) {
	t, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	a := &AssertStmt{Test: t}
	if p.tok.Kind == COMMA {
		p.next()
		m, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		a.Msg = m
	}
	return a, nil
}

func (p *Parser) parseGlobalStmt() (Stmt, error) {
	ns, err := p.parseNameList()
	if err != nil {
		return nil, err
	}
	return &GlobalStmt{Names: ns}, nil
}

func (p *Parser) parseNonlocalStmt() (Stmt, error) {
	ns, err := p.parseNameList()
	if err != nil {
		return nil, err
	}
	return &NonlocalStmt{Names: ns}, nil
}

func (p *Parser) parseNameList() ([]string, error) {
	var ns []string
	for {
		if p.tok.Kind != IDENT {
			return nil, p.errf("expected identifier")
		}
		ns = append(ns, p.tok.Lit)
		p.next()
		if p.tok.Kind != COMMA {
			break
		}
		p.next()
	}
	return ns, nil
}

func (p *Parser) parseImportStmt() (Stmt, error) {
	var names []Alias
	for {
		if p.tok.Kind != IDENT {
			return nil, p.errf("expected module name")
		}
		n := p.tok.Lit
		p.next()
		a := Alias{Name: n}
		if p.tok.Kind == KwAs {
			p.next()
			if p.tok.Kind != IDENT {
				return nil, p.errf("expected identifier after as")
			}
			a.As = p.tok.Lit
			p.next()
		}
		names = append(names, a)
		if p.tok.Kind != COMMA {
			break
		}
		p.next()
	}
	return &ImportStmt{Names: names}, nil
}

func (p *Parser) parseImportFromStmt() (Stmt, error) {
	p.next() // from
	rel := 0
	for p.tok.Kind == DOT {
		rel++
		p.next()
	}
	mod := ""
	if p.tok.Kind == IDENT {
		mod = p.tok.Lit
		p.next()
		for p.tok.Kind == DOT {
			mod += "."
			p.next()
			if p.tok.Kind != IDENT {
				return nil, p.errf("expected identifier in module path")
			}
			mod += p.tok.Lit
			p.next()
		}
	}
	if err := p.want(KwImport); err != nil {
		return nil, err
	}
	st := &ImportFrom{Module: mod, Relative: rel}
	if p.tok.Kind == STAR {
		p.next()
		st.Star = true
		return st, nil
	}
	if p.tok.Kind == LPAREN {
		p.next()
		for p.tok.Kind != RPAREN {
			al, err := p.parseImportAlias()
			if err != nil {
				return nil, err
			}
			st.Names = append(st.Names, al)
			if p.tok.Kind == COMMA {
				p.next()
			}
		}
		p.next()
		return st, nil
	}
	for {
		al, err := p.parseImportAlias()
		if err != nil {
			return nil, err
		}
		st.Names = append(st.Names, al)
		if p.tok.Kind != COMMA {
			break
		}
		p.next()
	}
	return st, nil
}

func (p *Parser) parseImportAlias() (Alias, error) {
	if p.tok.Kind != IDENT {
		return Alias{}, p.errf("expected name")
	}
	n := p.tok.Lit
	p.next()
	a := Alias{Name: n}
	if p.tok.Kind == KwAs {
		p.next()
		if p.tok.Kind != IDENT {
			return Alias{}, p.errf("expected identifier after as")
		}
		a.As = p.tok.Lit
		p.next()
	}
	return a, nil
}

func (p *Parser) parseAssignOrExprStmt() (Stmt, error) {
	lhs, err := p.parseExprList()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == COLON {
		p.next()
		return p.parseAnnAssign(lhs)
	}
	if isAugAssign(p.tok.Kind) {
		op := p.tok.Kind
		p.next()
		rhs, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return &AssignStmt{Targets: lhs, Op: op, Value: rhs}, nil
	}
	if p.tok.Kind == EQ {
		p.next()
		rhs, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return &AssignStmt{Targets: lhs, Op: EQ, Value: rhs}, nil
	}
	if len(lhs) != 1 {
		return nil, p.errf("invalid syntax in assignment")
	}
	return &ExprStmt{Value: lhs[0]}, nil
}

func (p *Parser) parseAnnAssign(lhs []Expr) (Stmt, error) {
	if len(lhs) != 1 {
		return nil, p.errf("annotated assignment needs single target")
	}
	ann, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	var val Expr
	if p.tok.Kind == EQ {
		p.next()
		val, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	return &AnnAssign{Target: lhs[0], Annotation: ann, Value: val}, nil
}

// parseLayoutFieldStmt parses `soa name: ann [= val]` / `aos ...` (Roma4D SoA/AoS field layout).
func (p *Parser) parseLayoutFieldStmt() ([]Stmt, error) {
	lay := p.tok.Kind
	p.next()
	lhs, err := p.parseExprList()
	if err != nil {
		return nil, err
	}
	if len(lhs) != 1 {
		return nil, p.errf("layout annotation expects a single target")
	}
	if p.tok.Kind != COLON {
		return nil, p.errf("expected ':' after field name")
	}
	p.next()
	ann, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	var val Expr
	if p.tok.Kind == EQ {
		p.next()
		val, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	a := &AnnAssign{Layout: &lay, Target: lhs[0], Annotation: ann, Value: val}
	if err := p.expectEndOfSimpleLine(); err != nil {
		return nil, err
	}
	return []Stmt{a}, nil
}

func isAugAssign(k TokenKind) bool {
	switch k {
	case PLUSEQ, MINUSEQ, STAREQ, SLASHEQ, DBLSLASHEQ, PERCENTEQ,
		AMPEQ, PIPEEQ, CARETEQ, LSHIFTEQ, RSHIFTEQ, DBLSTAREQ, ATEQ:
		return true
	default:
		return false
	}
}
