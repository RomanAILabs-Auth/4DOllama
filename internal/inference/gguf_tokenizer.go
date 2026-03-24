package inference

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"
)

const ggufMagic = "GGUF"

const (
	ggufTypeString uint32 = 8
	ggufTypeArray  uint32 = 9
)

// LoadGGUFTokenStrings reads tokenizer.ggml.tokens from a GGUF file (metadata only).
func LoadGGUFTokenStrings(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, err
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("not a GGUF file")
	}
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version < 2 || version > 4 {
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}
	var tensorCount, kvCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return nil, err
	}

	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, err
		}
		var typ uint32
		if err := binary.Read(f, binary.LittleEndian, &typ); err != nil {
			return nil, err
		}
		if key == "tokenizer.ggml.tokens" && typ == ggufTypeArray {
			var et uint32
			if err := binary.Read(f, binary.LittleEndian, &et); err != nil {
				return nil, err
			}
			var ne uint64
			if err := binary.Read(f, binary.LittleEndian, &ne); err != nil {
				return nil, err
			}
			if et != ggufTypeString {
				return nil, fmt.Errorf("tokenizer.ggml.tokens: expected string array, got type %d", et)
			}
			if ne > 1<<22 {
				return nil, fmt.Errorf("tokenizer.ggml.tokens: unreasonably large count %d", ne)
			}
			out := make([]string, 0, ne)
			for j := uint64(0); j < ne; j++ {
				s, err := readGGUFString(f)
				if err != nil {
					return nil, err
				}
				out = append(out, s)
			}
			if len(out) == 0 {
				return nil, fmt.Errorf("empty tokenizer.ggml.tokens")
			}
			return out, nil
		}
		if err := skipGGUFValue(f, typ); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF")
}

func readGGUFString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<28 {
		return "", fmt.Errorf("GGUF string too long")
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func skipGGUFValue(r io.Reader, typ uint32) error {
	switch typ {
	case 0:
		var x uint8
		return binary.Read(r, binary.LittleEndian, &x)
	case 1:
		var x int8
		return binary.Read(r, binary.LittleEndian, &x)
	case 2:
		var x uint16
		return binary.Read(r, binary.LittleEndian, &x)
	case 3:
		var x int16
		return binary.Read(r, binary.LittleEndian, &x)
	case 4:
		var x uint32
		return binary.Read(r, binary.LittleEndian, &x)
	case 5:
		var x int32
		return binary.Read(r, binary.LittleEndian, &x)
	case 6:
		var x float32
		return binary.Read(r, binary.LittleEndian, &x)
	case 7:
		var x uint8
		return binary.Read(r, binary.LittleEndian, &x)
	case 8:
		_, err := readGGUFString(r)
		return err
	case 9:
		var et uint32
		if err := binary.Read(r, binary.LittleEndian, &et); err != nil {
			return err
		}
		var ne uint64
		if err := binary.Read(r, binary.LittleEndian, &ne); err != nil {
			return err
		}
		for i := uint64(0); i < ne; i++ {
			if err := skipGGUFValue(r, et); err != nil {
				return err
			}
		}
		return nil
	case 10:
		var x uint64
		return binary.Read(r, binary.LittleEndian, &x)
	case 11:
		var x int64
		return binary.Read(r, binary.LittleEndian, &x)
	case 12:
		var x float64
		return binary.Read(r, binary.LittleEndian, &x)
	default:
		return fmt.Errorf("unknown GGUF value type %d", typ)
	}
}

const spPieceRune = '\u2581' // ▁ word-boundary marker in many GGUF tokenizers

// AppendSPPiece appends one SentencePiece-style token to the running decoded text.
func AppendSPPiece(b *strings.Builder, piece string) {
	if b.Len() == 0 {
		r, w := utf8.DecodeRuneInString(piece)
		if r == spPieceRune && w < len(piece) {
			b.WriteString(piece[w:])
		} else {
			b.WriteString(piece)
		}
		return
	}
	r, w := utf8.DecodeRuneInString(piece)
	if r == spPieceRune {
		b.WriteByte(' ')
		b.WriteString(piece[w:])
		return
	}
	b.WriteString(piece)
}

// PromptTokenBias nudges logits toward tokenizer pieces that appear in the prompt (readable continuation).
func PromptTokenBias(logits []float32, tokens []string, prompt string) {
	if len(logits) != len(tokens) || prompt == "" {
		return
	}
	pl := strings.ToLower(prompt)
	for i, t := range tokens {
		if i >= len(logits) {
			break
		}
		tl := strings.ToLower(t)
		tl = strings.TrimPrefix(tl, string(spPieceRune))
		if len(tl) >= 2 && strings.Contains(pl, tl) {
			logits[i] += 1.5
		}
	}
}
