package inference

import (
	"fmt"
	"path/filepath"
	"strings"
)

const stubVocabSize = 8192

// stubSentencePieceVocab builds a deterministic pseudo–SentencePiece table for native .4dai runs
// when no GGUF tokenizer is present.
func stubSentencePieceVocab() []string {
	toks := make([]string, stubVocabSize)
	for i := range toks {
		toks[i] = fmt.Sprintf("▁%d", i)
	}
	return toks
}

// LoadInferenceTokenizerVocab loads tokenizer strings for stub autoreg.
// If tokenizerGGUF is non-empty and points at a GGUF with tokenizer.ggml.tokens, that vocabulary is used
// (correct detokenization for converted .4dai models). Otherwise .gguf weights load their own tokenizer;
// pure .4dai without sidecar falls back to a small stub table.
func LoadInferenceTokenizerVocab(modelPath, tokenizerGGUF string) ([]string, error) {
	modelPath = strings.TrimSpace(modelPath)
	tokenizerGGUF = strings.TrimSpace(tokenizerGGUF)
	if tokenizerGGUF != "" {
		if toks, err := LoadGGUFTokenStrings(tokenizerGGUF); err == nil && len(toks) > 0 {
			return toks, nil
		}
	}
	if modelPath == "" {
		return nil, fmt.Errorf("model path required for tokenizer")
	}
	ext := strings.ToLower(filepath.Ext(modelPath))
	if ext == ".gguf" {
		return LoadGGUFTokenStrings(modelPath)
	}
	if ext == ".4dai" || ext == ".multi4dai" {
		return stubSentencePieceVocab(), nil
	}
	if toks, err := LoadGGUFTokenStrings(modelPath); err == nil && len(toks) > 0 {
		return toks, nil
	}
	return stubSentencePieceVocab(), nil
}
