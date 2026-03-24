package ollamareg

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ImportFromOllamaHome copies the GGUF model blob from an existing Ollama install
// (after `ollama pull model:tag`) into destDir as "{FileStem}.gguf".
func ImportFromOllamaHome(rawRef, destDir, ollamaModelsDir string, prog io.Writer) (outPath string, err error) {
	ref, err := ParseRef(rawRef)
	if err != nil {
		return "", err
	}
	_ = os.MkdirAll(destDir, 0o755)

	man := filepath.Join(ollamaModelsDir, "manifests", ref.Host, ref.Namespace, ref.Model, ref.Tag)
	b, err := os.ReadFile(man)
	if err != nil {
		return "", fmt.Errorf("read ollama manifest %s: %w (run `ollama pull %s` first)", man, err, ref.Display())
	}

	var mf registryManifest
	if err := json.Unmarshal(b, &mf); err != nil {
		return "", fmt.Errorf("parse ollama manifest: %w", err)
	}

	var best string
	var bestSize int64
	for _, L := range mf.Layers {
		if L.MediaType == mediaTypeModel && L.Digest != "" {
			if L.Size >= bestSize {
				bestSize = L.Size
				best = L.Digest
			}
		}
	}
	if best == "" {
		return "", fmt.Errorf("%w: Ollama manifest at %s has no %s layer", ErrNoGGUFLayer, man, mediaTypeModel)
	}

	want, err := parseDigest(best)
	if err != nil {
		return "", err
	}
	blobName := strings.ReplaceAll(best, ":", "-")
	src := filepath.Join(ollamaModelsDir, "blobs", blobName)
	if _, err := os.Stat(src); err != nil {
		// alternate layout
		alt := filepath.Join(ollamaModelsDir, "blobs", "sha256-"+want)
		if st, e2 := os.Stat(alt); e2 == nil {
			src = alt
			_ = st
		} else {
			return "", fmt.Errorf("blob not found: %s (%v)", src, err)
		}
	}

	outPath = filepath.Join(destDir, ref.FileStem()+".gguf")
	if prog != nil {
		_, _ = fmt.Fprintf(prog, "importing %s → %s\n", src, outPath)
	}
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()
	out, err := os.Create(outPath)
	if err != nil {
		return "", err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		_ = os.Remove(outPath)
		return "", err
	}
	return outPath, nil
}
