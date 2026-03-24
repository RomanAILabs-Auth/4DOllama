package ollamareg

import "errors"

var (
	errEmptyRef     = errors.New("empty model reference")
	ErrUnauthorized = errors.New("registry returned 401 — set OLLAMA_REGISTRY or pull with the official ollama CLI, then use import-ollama")
	ErrNoGGUFLayer  = errors.New("manifest has no application/vnd.ollama.image.model layer (tensor-split models need `ollama pull` + import-ollama)")
)
