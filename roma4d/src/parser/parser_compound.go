package parser

// parseCompoundStmt parses one compound statement.
func (p *Parser) parseCompoundStmt() (Stmt, error) {
	if p.tok.Kind == KwAsync {
		p.next()
		switch p.tok.Kind {
		case KwDef:
			return p.parseFuncDef(true)
		case KwFor:
			return p.parseForStmt(true, false)
		case KwWith:
			return p.parseWithStmt(true)
		case KwPar:
			p.next()
			if err := p.want(KwFor); err != nil {
				return nil, err
			}
			return p.parseForRest(true, true)
		default:
			return nil, p.errf("invalid async statement")
		}
	}
	switch p.tok.Kind {
	case KwDef:
		return p.parseFuncDef(false)
	case KwClass:
		return p.parseClassDef()
	case KwIf:
		return p.parseIfStmt()
	case KwWhile:
		return p.parseWhileStmt()
	case KwFor:
		return p.parseForStmt(false, false)
	case KwPar:
		p.next()
		if err := p.want(KwFor); err != nil {
			return nil, err
		}
		return p.parseForRest(false, true)
	case KwTry:
		return p.parseTryStmt()
	case KwWith:
		return p.parseWithStmt(false)
	case KwMatch:
		return p.parseMatchStmt()
	case KwType:
		return p.parseTypeStmt()
	case KwUnsafe:
		return p.parseUnsafeStmt()
	case KwSpacetime:
		return p.parseSpacetimeStmt()
	default:
		return nil, p.errf("not a compound statement")
	}
}

func (p *Parser) parseSuite() ([]Stmt, error) {
	if p.tok.Kind == NEWLINE {
		p.next()
		// Blank and comment-only lines do not produce INDENT; skip their NEWLINEs.
		for p.tok.Kind == NEWLINE {
			p.next()
		}
		if p.tok.Kind != INDENT {
			return nil, p.errf("expected an indented block after ':'")
		}
		p.next()
		return p.parseStmtList(true)
	}
	return p.parseSimpleStmts()
}

func (p *Parser) parseDecorated() ([]Stmt, error) {
	var decos []Expr
	for p.tok.Kind == AT {
		p.next()
		d, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		decos = append(decos, d)
		if p.tok.Kind == NEWLINE {
			p.next()
		} else {
			return nil, p.errf("expected newline after decorator")
		}
	}
	if p.tok.Kind != KwDef && p.tok.Kind != KwClass {
		return nil, p.errf("decorator must be followed by def or class")
	}
	var s Stmt
	var err error
	if p.tok.Kind == KwDef {
		s, err = p.parseFuncDef(false)
	} else {
		s, err = p.parseClassDef()
	}
	if err != nil {
		return nil, err
	}
	switch t := s.(type) {
	case *FunctionDef:
		t.Decorators = append(decos, t.Decorators...)
	case *ClassDef:
		t.Decorators = append(decos, t.Decorators...)
	}
	if err := p.expectEndOfSimpleLine(); err != nil {
		return nil, err
	}
	return []Stmt{s}, nil
}

func (p *Parser) parseFuncDef(async bool) (Stmt, error) {
	if err := p.want(KwDef); err != nil {
		return nil, err
	}
	if p.tok.Kind != IDENT {
		return nil, p.errf("expected function name")
	}
	name := p.tok.Lit
	p.next()
	if err := p.want(LPAREN); err != nil {
		return nil, err
	}
	args, err := p.parseArguments(false)
	if err != nil {
		return nil, err
	}
	if err := p.want(RPAREN); err != nil {
		return nil, err
	}
	var ret Expr
	if p.tok.Kind == ARROW {
		p.next()
		ret, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	return &FunctionDef{Async: async, Name: name, Args: args, Returns: ret, Body: body}, nil
}

func (p *Parser) parseClassDef() (Stmt, error) {
	if err := p.want(KwClass); err != nil {
		return nil, err
	}
	if p.tok.Kind != IDENT {
		return nil, p.errf("expected class name")
	}
	name := p.tok.Lit
	p.next()
	var bases []Expr
	var kws []*Keyword
	if p.tok.Kind == LPAREN {
		p.next()
		if p.tok.Kind != RPAREN {
			for {
				if p.tok.Kind == STAR || p.tok.Kind == DBLSTAR {
					return nil, p.errf("unsupported * in class bases")
				}
				e, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				if p.tok.Kind == EQ {
					p.next()
					if nm, ok := e.(*Name); ok {
						v, err := p.parseExpr()
						if err != nil {
							return nil, err
						}
						kws = append(kws, &Keyword{Name: nm.Id, Val: v})
					} else {
						return nil, p.errf("keyword argument must be a name")
					}
				} else {
					bases = append(bases, e)
				}
				if p.tok.Kind == COMMA {
					p.next()
					continue
				}
				break
			}
		}
		if err := p.want(RPAREN); err != nil {
			return nil, err
		}
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	return &ClassDef{Name: name, Bases: bases, Keywords: kws, Body: body}, nil
}

func (p *Parser) parseIfStmt() (Stmt, error) {
	if err := p.want(KwIf); err != nil {
		return nil, err
	}
	test, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	st := &IfStmt{Test: test, Body: body}
	for p.tok.Kind == KwElif {
		p.next()
		t, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		b, err := p.parseSuite()
		if err != nil {
			return nil, err
		}
		st.Elifs = append(st.Elifs, ElifClause{Test: t, Body: b})
	}
	if p.tok.Kind == KwElse {
		p.next()
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		st.Else, err = p.parseSuite()
		if err != nil {
			return nil, err
		}
	}
	return st, nil
}

func (p *Parser) parseWhileStmt() (Stmt, error) {
	if err := p.want(KwWhile); err != nil {
		return nil, err
	}
	test, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	w := &WhileStmt{Test: test, Body: body}
	if p.tok.Kind == KwElse {
		p.next()
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		w.Else, err = p.parseSuite()
		if err != nil {
			return nil, err
		}
	}
	return w, nil
}

func (p *Parser) parseForStmt(async, par bool) (Stmt, error) {
	if err := p.want(KwFor); err != nil {
		return nil, err
	}
	return p.parseForRest(async, par)
}

func (p *Parser) parseForRest(async, par bool) (Stmt, error) {
	tgt, err := p.parseExprListNoCompare()
	if err != nil {
		return nil, err
	}
	if len(tgt) != 1 {
		return nil, p.errf("multiple targets in for need tuple unpacking (simplified parser)")
	}
	if err := p.want(KwIn); err != nil {
		return nil, err
	}
	it, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	var els []Stmt
	if p.tok.Kind == KwElse {
		p.next()
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		els, err = p.parseSuite()
		if err != nil {
			return nil, err
		}
	}
	if par {
		return &ParForStmt{Async: async, Target: tgt[0], Iter: it, Body: body, Else: els}, nil
	}
	return &ForStmt{Async: async, Target: tgt[0], Iter: it, Body: body, Else: els}, nil
}

func (p *Parser) parseTryStmt() (Stmt, error) {
	if err := p.want(KwTry); err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	tr := &TryStmt{Body: body}
	for p.tok.Kind == KwExcept {
		p.next()
		var typ Expr
		var nm string
		if p.tok.Kind != COLON {
			typ, err = p.parseExpr()
			if err != nil {
				return nil, err
			}
			if p.tok.Kind == KwAs {
				p.next()
				if p.tok.Kind != IDENT {
					return nil, p.errf("expected name after as")
				}
				nm = p.tok.Lit
				p.next()
			}
		}
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		hbody, err := p.parseSuite()
		if err != nil {
			return nil, err
		}
		tr.Handlers = append(tr.Handlers, ExceptHandler{Type: typ, Name: nm, Body: hbody})
	}
	if p.tok.Kind == KwElse {
		p.next()
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		tr.Else, err = p.parseSuite()
		if err != nil {
			return nil, err
		}
	}
	if p.tok.Kind == KwFinally {
		p.next()
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		tr.Finally, err = p.parseSuite()
		if err != nil {
			return nil, err
		}
	}
	return tr, nil
}

func (p *Parser) parseWithStmt(async bool) (Stmt, error) {
	if err := p.want(KwWith); err != nil {
		return nil, err
	}
	var items []WithItem
	for {
		ctx, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		wi := WithItem{Context: ctx}
		if p.tok.Kind == KwAs {
			p.next()
			t, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			wi.Target = t
		}
		items = append(items, wi)
		if p.tok.Kind == COMMA {
			p.next()
			continue
		}
		break
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	_ = async // future: mark WithStmt async
	return &WithStmt{Items: items, Body: body}, nil
}

func (p *Parser) parseUnsafeStmt() (Stmt, error) {
	if err := p.want(KwUnsafe); err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	return &UnsafeStmt{Body: body}, nil
}

func (p *Parser) parseSpacetimeStmt() (Stmt, error) {
	if err := p.want(KwSpacetime); err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseSuite()
	if err != nil {
		return nil, err
	}
	return &SpacetimeStmt{Body: body}, nil
}

func (p *Parser) parseMatchStmt() (Stmt, error) {
	if err := p.want(KwMatch); err != nil {
		return nil, err
	}
	subj, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	if err := p.want(NEWLINE); err != nil {
		return nil, err
	}
	if err := p.want(INDENT); err != nil {
		return nil, err
	}
	var cases []MatchCase
	for p.tok.Kind == KwCase {
		p.next()
		pat, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		var guard Expr
		if p.tok.Kind == KwIf {
			p.next()
			guard, err = p.parseExpr()
			if err != nil {
				return nil, err
			}
		}
		if err := p.want(COLON); err != nil {
			return nil, err
		}
		b, err := p.parseSuite()
		if err != nil {
			return nil, err
		}
		cases = append(cases, MatchCase{Pattern: pat, Guard: guard, Body: b})
	}
	if err := p.want(DEDENT); err != nil {
		return nil, err
	}
	return &MatchStmt{Subject: subj, Cases: cases}, nil
}

func (p *Parser) parseTypeStmt() (Stmt, error) {
	if err := p.want(KwType); err != nil {
		return nil, err
	}
	if p.tok.Kind != IDENT {
		return nil, p.errf("expected type alias name")
	}
	n := p.tok.Lit
	p.next()
	if err := p.want(EQ); err != nil {
		return nil, err
	}
	v, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &TypeAliasStmt{Name: n, Value: v}, nil
}

func (p *Parser) parseArguments(isLambda bool) (*Arguments, error) {
	a := &Arguments{}
	for p.tok.Kind != RPAREN && p.tok.Kind != EOF {
		if p.tok.Kind == STAR {
			p.next()
			if p.tok.Kind == IDENT {
				a.Vararg = p.tok.Lit
				p.next()
			}
			if p.tok.Kind == COMMA {
				p.next()
			}
			for p.tok.Kind != RPAREN && p.tok.Kind != EOF {
				if p.tok.Kind == DBLSTAR {
					p.next()
					if p.tok.Kind != IDENT {
						return nil, p.errf("expected name after **")
					}
					a.Kwarg = p.tok.Lit
					p.next()
					break
				}
				arg, err := p.parseArgDef(isLambda)
				if err != nil {
					return nil, err
				}
				a.Kwonly = append(a.Kwonly, arg)
				if p.tok.Kind == COMMA {
					p.next()
				}
			}
			break
		}
		if p.tok.Kind == DBLSTAR {
			p.next()
			if p.tok.Kind != IDENT {
				return nil, p.errf("expected name after **")
			}
			a.Kwarg = p.tok.Lit
			p.next()
			break
		}
		arg, err := p.parseArgDef(isLambda)
		if err != nil {
			return nil, err
		}
		a.Args = append(a.Args, arg)
		if p.tok.Kind == COMMA {
			p.next()
			continue
		}
		break
	}
	return a, nil
}

func (p *Parser) parseArgDef(isLambda bool) (*Arg, error) {
	if p.tok.Kind != IDENT {
		return nil, p.errf("expected parameter name")
	}
	nm := p.tok.Lit
	p.next()
	ar := &Arg{Name: nm}
	if p.tok.Kind == COLON {
		p.next()
		an, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		ar.Annot = an
	}
	if p.tok.Kind == EQ {
		if isLambda {
			return nil, p.errf("default in lambda not supported here")
		}
		p.next()
		df, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		ar.Default = df
	}
	return ar, nil
}
