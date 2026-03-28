package parser

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Lexer tokenizes Roma4D / Python 3.12 source with INDENT/DEDENT and r4d keywords.
type Lexer struct {
	src  []rune
	path string
	i    int
	line int
	col  int

	indent []int
	bol    bool
	depth  int // nesting of (), [], {}

	queue []Token
}

// NewLexer returns a lexer over a complete source file. path is used in diagnostics only.
func NewLexer(path, src string) *Lexer {
	return &Lexer{
		src:    []rune(src),
		path:   path,
		line:   1,
		col:    1,
		indent: []int{0},
		bol:    true,
	}
}

// Next returns the next token, including synthetic INDENT/DEDENT.
func (l *Lexer) Next() Token {
	for {
		if len(l.queue) > 0 {
			t := l.queue[0]
			l.queue = l.queue[1:]
			return t
		}
		if l.bol && l.depth == 0 {
			if err := l.processIndentLine(); err != nil {
				return l.illegal(err.Error())
			}
			if len(l.queue) > 0 {
				continue
			}
		}
		if l.i >= len(l.src) {
			l.flushDedentsAtEOF()
			if len(l.queue) > 0 {
				continue
			}
			return Token{Kind: EOF, Line: l.line, Col: l.col}
		}
		l.skipHorizontalSpace()
		if l.i >= len(l.src) {
			continue
		}
		switch l.src[l.i] {
		case '#':
			l.skipComment()
			continue
		case '\n', '\r':
			l.consumeEOL()
			continue
		case '\\':
			if l.i+1 < len(l.src) && l.src[l.i+1] == '\n' {
				l.i += 2
				l.line++
				l.col = 1
				continue
			}
		}
		l.bol = false
		return l.lexNonSpace()
	}
}

func (l *Lexer) flushDedentsAtEOF() {
	for len(l.indent) > 1 {
		l.indent = l.indent[:len(l.indent)-1]
		l.queue = append(l.queue, Token{Kind: DEDENT, Line: l.line, Col: l.col})
	}
}

func (l *Lexer) illegal(msg string) Token {
	return Token{Kind: ILLEGAL, Lit: msg, Line: l.line, Col: l.col}
}

func (l *Lexer) processIndentLine() error {
	startLine := l.line
	startCol := l.col
	// Blank or comment-only logical line: no indent tokens.
	if l.onlyWhitespaceOrCommentLine() {
		l.skipToEndOfLine()
		if l.i < len(l.src) {
			l.consumeEOL()
		}
		return nil
	}
	w, err := l.measureIndentAtBOL()
	if err != nil {
		return fmt.Errorf("%s:%d:%d: %w", l.path, startLine, startCol, err)
	}
	top := l.indent[len(l.indent)-1]
	switch {
	case w > top:
		l.indent = append(l.indent, w)
		l.queue = append(l.queue, Token{Kind: INDENT, Line: startLine, Col: 1})
	case w < top:
		for len(l.indent) > 1 && l.indent[len(l.indent)-1] > w {
			l.indent = l.indent[:len(l.indent)-1]
			l.queue = append(l.queue, Token{Kind: DEDENT, Line: startLine, Col: 1})
		}
		if l.indent[len(l.indent)-1] != w {
			return fmt.Errorf("unindent does not match any outer indentation level")
		}
	}
	// Indentation for this logical line is resolved; do not re-run processIndent until the next newline.
	l.bol = false
	return nil
}

func (l *Lexer) onlyWhitespaceOrCommentLine() bool {
	j := l.i
	for j < len(l.src) {
		switch l.src[j] {
		case ' ', '\t':
			j++
		case '#':
			return true
		case '\n', '\r':
			return true
		default:
			return false
		}
	}
	return true
}

func (l *Lexer) skipToEndOfLine() {
	for l.i < len(l.src) && l.src[l.i] != '\n' && l.src[l.i] != '\r' {
		l.advance()
	}
}

func (l *Lexer) measureIndentAtBOL() (int, error) {
	var width int
	seenSpace, seenTab := false, false
	start := l.i
	for l.i < len(l.src) {
		c := l.src[l.i]
		switch c {
		case ' ':
			seenSpace = true
			width++
			l.advance()
		case '\t':
			seenTab = true
			width = (width + 8) &^ 7
			l.advance()
		case '#', '\n', '\r':
			return 0, nil // caller shouldn't hit — onlyWhitespace handled
		default:
			if seenSpace && seenTab {
				return 0, fmt.Errorf("inconsistent use of tabs and spaces in indentation")
			}
			_ = start
			return width, nil
		}
	}
	return width, nil
}

func (l *Lexer) skipHorizontalSpace() {
	for l.i < len(l.src) {
		c := l.src[l.i]
		if c == ' ' || c == '\t' {
			l.advance()
			continue
		}
		break
	}
}

func (l *Lexer) skipComment() {
	for l.i < len(l.src) && l.src[l.i] != '\n' {
		l.advance()
	}
}

func (l *Lexer) consumeEOL() {
	if l.src[l.i] == '\r' {
		l.advance()
		if l.i < len(l.src) && l.src[l.i] == '\n' {
			l.advance()
		}
	} else if l.src[l.i] == '\n' {
		l.advance()
	}
	l.line++
	l.col = 1
	l.bol = true
	if l.depth == 0 {
		l.queue = append(l.queue, Token{Kind: NEWLINE, Lit: "\\n", Line: l.line - 1, Col: 1})
	}
}

func (l *Lexer) advance() {
	if l.i >= len(l.src) {
		return
	}
	ch := l.src[l.i]
	l.i++
	if ch == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
}

func (l *Lexer) peek() rune {
	if l.i >= len(l.src) {
		return 0
	}
	return l.src[l.i]
}

func (l *Lexer) lexNonSpace() Token {
	line, col := l.line, l.col
	ch := l.src[l.i]

	openClose := func(kind TokenKind, d int) Token {
		l.advance()
		l.depth += d
		if l.depth < 0 {
			l.depth = 0
		}
		return Token{Kind: kind, Lit: string(ch), Line: line, Col: col}
	}

	switch ch {
	case '(':
		return openClose(LPAREN, 1)
	case ')':
		return openClose(RPAREN, -1)
	case '[':
		return openClose(LBRACK, 1)
	case ']':
		return openClose(RBRACK, -1)
	case '{':
		return openClose(LBRACE, 1)
	case '}':
		return openClose(RBRACE, -1)
	case ':':
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: COLONEQ, Lit: ":=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: COLON, Lit: ":", Line: line, Col: col}
	case ',':
		l.advance()
		return Token{Kind: COMMA, Lit: ",", Line: line, Col: col}
	case ';':
		l.advance()
		return Token{Kind: SEMI, Lit: ";", Line: line, Col: col}
	case '@':
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: ATEQ, Lit: "@=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: AT, Lit: "@", Line: line, Col: col}
	case '~':
		l.advance()
		return Token{Kind: TILDE, Lit: "~", Line: line, Col: col}
	}

	if ch == '.' && l.i+2 < len(l.src) && l.src[l.i+1] == '.' && l.src[l.i+2] == '.' {
		l.i += 3
		l.col += 3
		return Token{Kind: ELLIPSIS, Lit: "...", Line: line, Col: col}
	}
	if ch == '.' {
		l.advance()
		return Token{Kind: DOT, Lit: ".", Line: line, Col: col}
	}

	// Multi-char operators
	if ch == '=' && l.peekNext(1) == '=' {
		l.advance2()
		return Token{Kind: EQEQ, Lit: "==", Line: line, Col: col}
	}
	if ch == '!' && l.peekNext(1) == '=' {
		l.advance2()
		return Token{Kind: NOTEQ, Lit: "!=", Line: line, Col: col}
	}
	if ch == '<' && l.peekNext(1) == '=' {
		l.advance2()
		return Token{Kind: LTE, Lit: "<=", Line: line, Col: col}
	}
	if ch == '>' && l.peekNext(1) == '=' {
		l.advance2()
		return Token{Kind: GTE, Lit: ">=", Line: line, Col: col}
	}
	if ch == '<' && l.peekNext(1) == '<' {
		if l.peekNext(2) == '=' {
			l.advance3()
			return Token{Kind: LSHIFTEQ, Lit: "<<=", Line: line, Col: col}
		}
		l.advance2()
		return Token{Kind: LSHIFT, Lit: "<<", Line: line, Col: col}
	}
	if ch == '>' && l.peekNext(1) == '>' {
		if l.peekNext(2) == '=' {
			l.advance3()
			return Token{Kind: RSHIFTEQ, Lit: ">>=", Line: line, Col: col}
		}
		l.advance2()
		return Token{Kind: RSHIFT, Lit: ">>", Line: line, Col: col}
	}
	if ch == '+' {
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: PLUSEQ, Lit: "+=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: PLUS, Lit: "+", Line: line, Col: col}
	}
	if ch == '-' {
		if l.peekNext(1) == '>' {
			l.advance2()
			return Token{Kind: ARROW, Lit: "->", Line: line, Col: col}
		}
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: MINUSEQ, Lit: "-=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: MINUS, Lit: "-", Line: line, Col: col}
	}
	if ch == '*' {
		if l.peekNext(1) == '*' {
			if l.peekNext(2) == '=' {
				l.advance3()
				return Token{Kind: DBLSTAREQ, Lit: "**=", Line: line, Col: col}
			}
			l.advance2()
			return Token{Kind: DBLSTAR, Lit: "**", Line: line, Col: col}
		}
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: STAREQ, Lit: "*=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: STAR, Lit: "*", Line: line, Col: col}
	}
	if ch == '/' {
		if l.peekNext(1) == '/' {
			if l.peekNext(2) == '=' {
				l.advance3()
				return Token{Kind: DBLSLASHEQ, Lit: "//=", Line: line, Col: col}
			}
			l.advance2()
			return Token{Kind: DBLSLASH, Lit: "//", Line: line, Col: col}
		}
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: SLASHEQ, Lit: "/=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: SLASH, Lit: "/", Line: line, Col: col}
	}
	if ch == '%' {
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: PERCENTEQ, Lit: "%=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: PERCENT, Lit: "%", Line: line, Col: col}
	}
	if ch == '&' {
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: AMPEQ, Lit: "&=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: AMP, Lit: "&", Line: line, Col: col}
	}
	if ch == '|' {
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: PIPEEQ, Lit: "|=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: PIPE, Lit: "|", Line: line, Col: col}
	}
	if ch == '^' {
		if l.peekNext(1) == '=' {
			l.advance2()
			return Token{Kind: CARETEQ, Lit: "^=", Line: line, Col: col}
		}
		l.advance()
		return Token{Kind: CARET, Lit: "^", Line: line, Col: col}
	}
	if ch == '=' {
		l.advance()
		return Token{Kind: EQ, Lit: "=", Line: line, Col: col}
	}
	if ch == '<' {
		l.advance()
		return Token{Kind: LT, Lit: "<", Line: line, Col: col}
	}
	if ch == '>' {
		l.advance()
		return Token{Kind: GT, Lit: ">", Line: line, Col: col}
	}
	if ch == '!' {
		l.advance()
		return Token{Kind: EXCL, Lit: "!", Line: line, Col: col}
	}
	if ch == '"' || ch == '\'' {
		return l.lexString(ch, line, col)
	}

	if unicode.IsDigit(ch) || (ch == '.' && l.i+1 < len(l.src) && unicode.IsDigit(l.src[l.i+1])) {
		return l.lexNumber(line, col)
	}

	if ch == '_' || unicode.IsLetter(ch) {
		return l.lexIdent(line, col)
	}

	l.advance()
	return Token{Kind: ILLEGAL, Lit: fmt.Sprintf("unexpected character %q", ch), Line: line, Col: col}
}

func (l *Lexer) peekNext(n int) rune {
	if l.i+n >= len(l.src) {
		return 0
	}
	return l.src[l.i+n]
}

func (l *Lexer) advance2() {
	l.advance()
	l.advance()
}

func (l *Lexer) advance3() {
	l.advance()
	l.advance()
	l.advance()
}

func (l *Lexer) lexIdent(line, col int) Token {
	start := l.i
	for l.i < len(l.src) {
		c := l.src[l.i]
		if c == '_' || unicode.IsLetter(c) || unicode.IsDigit(c) {
			l.advance()
			continue
		}
		break
	}
	lit := string(l.src[start:l.i])
	if kind, ok := KeywordTable[lit]; ok {
		return Token{Kind: kind, Lit: lit, Line: line, Col: col}
	}
	return Token{Kind: IDENT, Lit: lit, Line: line, Col: col}
}

func (l *Lexer) lexNumber(line, col int) Token {
	start := l.i
	base := 10
	if l.src[l.i] == '0' && l.i+1 < len(l.src) {
		switch l.src[l.i+1] {
		case 'x', 'X':
			base = 16
			l.advance()
			l.advance()
		case 'b', 'B':
			base = 2
			l.advance()
			l.advance()
		case 'o', 'O':
			base = 8
			l.advance()
			l.advance()
		}
	}
	for l.i < len(l.src) {
		c := l.src[l.i]
		if (base == 10 || base == 16) && c == '_' {
			l.advance()
			continue
		}
		if base == 16 && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			l.advance()
			continue
		}
		if (base == 10 || base == 8) && c >= '0' && c <= '9' {
			l.advance()
			continue
		}
		if base == 2 && (c == '0' || c == '1') {
			l.advance()
			continue
		}
		if base == 8 && c >= '0' && c <= '7' {
			l.advance()
			continue
		}
		break
	}
	// float / imag
	if base == 10 && l.i < len(l.src) && l.src[l.i] == '.' {
		l.advance()
		for l.i < len(l.src) && ((l.src[l.i] >= '0' && l.src[l.i] <= '9') || l.src[l.i] == '_') {
			l.advance()
		}
	}
	if base == 10 && l.i < len(l.src) && (l.src[l.i] == 'e' || l.src[l.i] == 'E') {
		l.advance()
		if l.i < len(l.src) && (l.src[l.i] == '+' || l.src[l.i] == '-') {
			l.advance()
		}
		for l.i < len(l.src) && l.src[l.i] >= '0' && l.src[l.i] <= '9' {
			l.advance()
		}
	}
	kind := INT
	lit := strings.ReplaceAll(string(l.src[start:l.i]), "_", "")
	if strings.ContainsAny(lit, ".eE") && base == 10 {
		kind = FLOAT
	}
	if l.i < len(l.src) && (l.src[l.i] == 'j' || l.src[l.i] == 'J') {
		l.advance()
		kind = IMAG
	}
	if kind == INT && base != 10 {
		if _, err := strconv.ParseInt(lit, 0, 64); err != nil {
			// keep literal text
		}
	}
	return Token{Kind: kind, Lit: lit, Line: line, Col: col}
}

func (l *Lexer) lexString(quote rune, line, col int) Token {
	delim := quote
	triple := false
	if l.i+2 < len(l.src) && l.src[l.i+1] == delim && l.src[l.i+2] == delim {
		triple = true
		l.advance()
		l.advance()
		l.advance()
	} else {
		l.advance()
	}
	var b strings.Builder
	for {
		if l.i >= len(l.src) {
			return Token{Kind: ILLEGAL, Lit: "unterminated string", Line: line, Col: col}
		}
		c := l.src[l.i]
		if c == '\\' && !triple {
			l.advance()
			if l.i >= len(l.src) {
				return Token{Kind: ILLEGAL, Lit: "unterminated escape", Line: line, Col: col}
			}
			switch l.src[l.i] {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case '\\', '"', '\'':
				b.WriteRune(l.src[l.i])
			default:
				b.WriteRune(l.src[l.i])
			}
			l.advance()
			continue
		}
		if !triple && c == '\n' {
			return Token{Kind: ILLEGAL, Lit: "EOL while scanning string literal", Line: line, Col: col}
		}
		if triple {
			if c == delim && l.i+2 < len(l.src) && l.src[l.i+1] == delim && l.src[l.i+2] == delim {
				l.advance()
				l.advance()
				l.advance()
				return Token{Kind: STRING, Lit: b.String(), Line: line, Col: col}
			}
		} else {
			if c == delim {
				l.advance()
				return Token{Kind: STRING, Lit: b.String(), Line: line, Col: col}
			}
		}
		b.WriteRune(c)
		l.advance()
	}
}

// PeekRunes exposes the underlying buffer for tests.
func (l *Lexer) atEOF() bool { return l.i >= len(l.src) }

func init() {
	_ = utf8.UTFMax
}
