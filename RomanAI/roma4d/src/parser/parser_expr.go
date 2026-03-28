package parser

// parseExprListNoCompare parses comma-separated expressions stopping at bitwise/or_expr level,
// so `for p in pos` does not treat `p in pos` as a single comparison (Python for-target rule).
func (p *Parser) parseExprListNoCompare() ([]Expr, error) {
	var es []Expr
	e, err := p.parseOrExpr()
	if err != nil {
		return nil, err
	}
	es = append(es, e)
	for p.tok.Kind == COMMA {
		p.next()
		if p.tok.Kind == NEWLINE || p.tok.Kind == COLON || p.tok.Kind == RPAREN || p.tok.Kind == RBRACK {
			break
		}
		e2, err := p.parseOrExpr()
		if err != nil {
			return nil, err
		}
		es = append(es, e2)
	}
	return es, nil
}

// parseExprList parses comma-separated expressions (no trailing comma required).
func (p *Parser) parseExprList() ([]Expr, error) {
	var es []Expr
	e, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	es = append(es, e)
	for p.tok.Kind == COMMA {
		p.next()
		if p.tok.Kind == NEWLINE || p.tok.Kind == COLON || p.tok.Kind == RPAREN || p.tok.Kind == RBRACK {
			break
		}
		e2, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		es = append(es, e2)
	}
	return es, nil
}

// parseExpr parses a full conditional expression.
func (p *Parser) parseExpr() (Expr, error) {
	return p.parseTest()
}

func (p *Parser) parseTest() (Expr, error) {
	body, err := p.parseOrTest()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == KwIf {
		p.next()
		tst, err := p.parseOrTest()
		if err != nil {
			return nil, err
		}
		if err := p.want(KwElse); err != nil {
			return nil, err
		}
		els, err := p.parseTest()
		if err != nil {
			return nil, err
		}
		return &IfExp{Test: tst, Body: body, Orelse: els}, nil
	}
	return body, nil
}

func (p *Parser) parseOrTest() (Expr, error) {
	x, err := p.parseAndTest()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind != KwOr {
		return x, nil
	}
	vals := []Expr{x}
	for p.tok.Kind == KwOr {
		p.next()
		y, err := p.parseAndTest()
		if err != nil {
			return nil, err
		}
		vals = append(vals, y)
	}
	return &BoolOp{Op: KwOr, Values: vals}, nil
}

func (p *Parser) parseAndTest() (Expr, error) {
	x, err := p.parseNotTest()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind != KwAnd {
		return x, nil
	}
	vals := []Expr{x}
	for p.tok.Kind == KwAnd {
		p.next()
		y, err := p.parseNotTest()
		if err != nil {
			return nil, err
		}
		vals = append(vals, y)
	}
	return &BoolOp{Op: KwAnd, Values: vals}, nil
}

func (p *Parser) parseNotTest() (Expr, error) {
	if p.tok.Kind == KwNot {
		p.next()
		x, err := p.parseNotTest()
		if err != nil {
			return nil, err
		}
		return &UnaryOp{Op: KwNot, Expr: x}, nil
	}
	return p.parseComparison()
}

func (p *Parser) parseComparison() (Expr, error) {
	x, err := p.parseOrExpr()
	if err != nil {
		return nil, err
	}
	var ops []TokenKind
	var comps []Expr
	for {
		var op TokenKind
		switch p.tok.Kind {
		case LT, GT, EQEQ, GTE, LTE, NOTEQ:
			op = p.tok.Kind
			p.next()
		case KwIn:
			op = KwIn
			p.next()
		case KwIs:
			op = KwIs
			p.next()
			if p.tok.Kind == KwNot {
				p.next()
			}
		default:
			if len(ops) == 0 {
				return x, nil
			}
			return &Compare{Left: x, Ops: ops, Comparators: comps}, nil
		}
		y, err := p.parseOrExpr()
		if err != nil {
			return nil, err
		}
		ops = append(ops, op)
		comps = append(comps, y)
	}
}

func (p *Parser) parseOrExpr() (Expr, error) {
	x, err := p.parseXorExpr()
	if err != nil {
		return nil, err
	}
	for p.tok.Kind == PIPE {
		op := p.tok.Kind
		p.next()
		y, err := p.parseXorExpr()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseXorExpr() (Expr, error) {
	x, err := p.parseAndExpr()
	if err != nil {
		return nil, err
	}
	for p.tok.Kind == CARET {
		op := p.tok.Kind
		p.next()
		y, err := p.parseAndExpr()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseAndExpr() (Expr, error) {
	x, err := p.parseShiftExpr()
	if err != nil {
		return nil, err
	}
	for p.tok.Kind == AMP {
		op := p.tok.Kind
		p.next()
		y, err := p.parseShiftExpr()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseShiftExpr() (Expr, error) {
	x, err := p.parseArithExpr()
	if err != nil {
		return nil, err
	}
	for p.tok.Kind == LSHIFT || p.tok.Kind == RSHIFT {
		op := p.tok.Kind
		p.next()
		y, err := p.parseArithExpr()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseArithExpr() (Expr, error) {
	x, err := p.parseTerm()
	if err != nil {
		return nil, err
	}
	for p.tok.Kind == PLUS || p.tok.Kind == MINUS {
		op := p.tok.Kind
		p.next()
		y, err := p.parseTerm()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseTerm() (Expr, error) {
	x, err := p.parseFactor()
	if err != nil {
		return nil, err
	}
	for {
		var op TokenKind
		switch p.tok.Kind {
		case STAR, AT, SLASH, DBLSLASH, PERCENT:
			op = p.tok.Kind
			p.next()
		default:
			return x, nil
		}
		y, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: op, Left: x, Right: y}
	}
}

func (p *Parser) parseFactor() (Expr, error) {
	switch p.tok.Kind {
	case PLUS, MINUS, TILDE:
		op := p.tok.Kind
		p.next()
		x, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		return &UnaryOp{Op: op, Expr: x}, nil
	default:
		return p.parsePower()
	}
}

func (p *Parser) parsePower() (Expr, error) {
	x, err := p.parseAtomExpr()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == DBLSTAR {
		p.next()
		y, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		x = &BinOp{Op: DBLSTAR, Left: x, Right: y}
	}
	return x, nil
}

func (p *Parser) parseAtomExpr() (Expr, error) {
	x, err := p.parseAtom()
	if err != nil {
		return nil, err
	}
	for {
		switch p.tok.Kind {
		case DOT:
			p.next()
			if p.tok.Kind != IDENT {
				return nil, p.errf("expected attribute name")
			}
			attr := p.tok.Lit
			p.next()
			x = &Attribute{Value: x, Attr: attr}
		case LPAREN:
			x, err = p.parseCallSuffix(x)
			if err != nil {
				return nil, err
			}
		case LBRACK:
			p.next()
			sl, err := p.parseSubscriptList()
			if err != nil {
				return nil, err
			}
			if err := p.want(RBRACK); err != nil {
				return nil, err
			}
			x = &Subscript{Value: x, Slice: sl}
		default:
			return x, nil
		}
	}
}

func (p *Parser) parseCallSuffix(fn Expr) (Expr, error) {
	if err := p.want(LPAREN); err != nil {
		return nil, err
	}
	call := &Call{Func: fn}
	if p.tok.Kind == RPAREN {
		p.next()
		return call, nil
	}
	for {
		if p.tok.Kind == STAR {
			p.next()
			v, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			call.Args = append(call.Args, &Starred{Value: v})
		} else if p.tok.Kind == KwFor {
			// generator: fn ( x for ... ) — rare; skip
			return nil, p.errf("generator expression in call not implemented")
		} else {
			// kwarg or positional
			arg, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			if p.tok.Kind == EQ && isNameLike(arg) {
				nm := arg.(*Name).Id
				p.next()
				v, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				call.Keywords = append(call.Keywords, &Keyword{Name: nm, Val: v})
			} else {
				call.Args = append(call.Args, arg)
			}
		}
		if p.tok.Kind == COMMA {
			p.next()
			if p.tok.Kind == RPAREN {
				break
			}
			continue
		}
		break
	}
	if err := p.want(RPAREN); err != nil {
		return nil, err
	}
	return call, nil
}

func isNameLike(e Expr) bool {
	_, ok := e.(*Name)
	return ok
}

func (p *Parser) parseSubscriptList() (Expr, error) {
	// Single slice or index; simplified
	if p.tok.Kind == COLON {
		p.next()
		return p.parseSliceAfterFirst(nil)
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == COLON {
		p.next()
		return p.parseSliceAfterFirst(first)
	}
	return first, nil
}

func (p *Parser) parseSliceAfterFirst(lower Expr) (Expr, error) {
	var up, step Expr
	if p.tok.Kind != COLON && p.tok.Kind != RBRACK {
		var err error
		up, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	if p.tok.Kind == COLON {
		p.next()
		if p.tok.Kind != RBRACK && p.tok.Kind != COMMA {
			var err error
			step, err = p.parseExpr()
			if err != nil {
				return nil, err
			}
		}
	}
	return &SliceExpr{Lower: lower, Upper: up, Step: step}, nil
}

func (p *Parser) parseAtom() (Expr, error) {
	switch p.tok.Kind {
	case LPAREN:
		return p.parseParenOrTuple()
	case LBRACK:
		return p.parseListDisplay()
	case LBRACE:
		return p.parseDictOrSet()
	case STRING, INT, FLOAT, IMAG:
		t := p.tok
		p.next()
		return &Constant{Kind: t.Kind, Text: t.Lit}, nil
	case KwTrue:
		p.next()
		return &Constant{Kind: KwTrue, Text: "True"}, nil
	case KwFalse:
		p.next()
		return &Constant{Kind: KwFalse, Text: "False"}, nil
	case KwNone:
		p.next()
		return &Constant{Kind: KwNone, Text: "None"}, nil
	case IDENT:
		t := p.tok
		n := t.Lit
		p.next()
		return &Name{Id: n, Line: t.Line, Col: t.Col}, nil
	case KwTime:
		t := p.tok
		line, col := t.Line, t.Col
		p.next()
		return &TimeCoord{Line: line, Col: col}, nil
	case KwVec4, KwRotor, KwMultivector, KwBorrow, KwMutborrow:
		t := p.tok
		n := t.Lit
		p.next()
		return &Name{Id: n, Line: t.Line, Col: t.Col}, nil
	case KwLambda:
		return p.parseLambda()
	case KwAwait:
		p.next()
		x, err := p.parseAtomExpr()
		if err != nil {
			return nil, err
		}
		return &AwaitExpr{Value: x}, nil
	default:
		return nil, p.errf("unexpected token %s in expression", p.tok.Lit)
	}
}

func (p *Parser) parseLambda() (Expr, error) {
	p.next() // lambda
	args := &Arguments{}
	for p.tok.Kind != COLON && p.tok.Kind != EOF {
		ar, err := p.parseArgDef(true)
		if err != nil {
			return nil, err
		}
		args.Args = append(args.Args, ar)
		if p.tok.Kind == COMMA {
			p.next()
			continue
		}
		break
	}
	if err := p.want(COLON); err != nil {
		return nil, err
	}
	body, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &Lambda{Args: args, Body: body}, nil
}

func (p *Parser) parseParenOrTuple() (Expr, error) {
	p.next() // (
	if p.tok.Kind == RPAREN {
		p.next()
		return &TupleExpr{Elts: nil}, nil
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == COMMA {
		elts := []Expr{first}
		for p.tok.Kind == COMMA {
			p.next()
			if p.tok.Kind == RPAREN {
				break
			}
			e, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			elts = append(elts, e)
		}
		if err := p.want(RPAREN); err != nil {
			return nil, err
		}
		return &TupleExpr{Elts: elts}, nil
	}
	if err := p.want(RPAREN); err != nil {
		return nil, err
	}
	return first, nil
}

func (p *Parser) parseListDisplay() (Expr, error) {
	p.next() // [
	if p.tok.Kind == RBRACK {
		p.next()
		return &ListExpr{}, nil
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == KwFor {
		return p.parseListComp(first)
	}
	elts := []Expr{first}
	for p.tok.Kind == COMMA {
		p.next()
		if p.tok.Kind == RBRACK {
			break
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elts = append(elts, e)
	}
	if err := p.want(RBRACK); err != nil {
		return nil, err
	}
	return &ListExpr{Elts: elts}, nil
}

func (p *Parser) parseListComp(elt Expr) (Expr, error) {
	p.next() // 'for'
	var comps []Comprehension
	for {
		tgt, err := p.parseOrExpr()
		if err != nil {
			return nil, err
		}
		if err := p.want(KwIn); err != nil {
			return nil, err
		}
		it, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		c := Comprehension{Target: tgt, Iter: it}
		for p.tok.Kind == KwIf {
			p.next()
			cond, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			c.Ifs = append(c.Ifs, cond)
		}
		comps = append(comps, c)
		if p.tok.Kind == KwFor {
			p.next()
			continue
		}
		break
	}
	if err := p.want(RBRACK); err != nil {
		return nil, err
	}
	return &ListComp{Elt: elt, Comps: comps}, nil
}

func (p *Parser) parseDictOrSet() (Expr, error) {
	p.next() // {
	if p.tok.Kind == RBRACE {
		p.next()
		return &DictExpr{}, nil
	}
	first, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.tok.Kind == COLON {
		p.next()
		val, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		keys := []Expr{first}
		vals := []Expr{val}
		for p.tok.Kind == COMMA {
			p.next()
			if p.tok.Kind == RBRACE {
				break
			}
			k, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			if err := p.want(COLON); err != nil {
				return nil, err
			}
			v, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			keys = append(keys, k)
			vals = append(vals, v)
		}
		if err := p.want(RBRACE); err != nil {
			return nil, err
		}
		return &DictExpr{Keys: keys, Values: vals}, nil
	}
	// set literal {a, b}
	elts := []Expr{first}
	for p.tok.Kind == COMMA {
		p.next()
		if p.tok.Kind == RBRACE {
			break
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elts = append(elts, e)
	}
	if err := p.want(RBRACE); err != nil {
		return nil, err
	}
	return &SetExpr{Elts: elts}, nil
}

func (p *Parser) parseStarExpr() (Expr, error) {
	if p.tok.Kind == STAR {
		p.next()
		x, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return &Starred{Value: x}, nil
	}
	return p.parseExpr()
}
