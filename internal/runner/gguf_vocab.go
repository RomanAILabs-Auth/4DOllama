package runner

import "github.com/4dollama/4dollama/internal/inference"

// LoadGGUFVocab loads tokenizer.ggml.tokens from a GGUF file (used by stub autoregressive detokenization).
func LoadGGUFVocab(modelPath string) ([]string, error) {
	return inference.LoadGGUFTokenStrings(modelPath)
}
