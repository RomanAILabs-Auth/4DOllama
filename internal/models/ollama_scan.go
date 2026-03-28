package models

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

const ollamaMediaTypeModel = "application/vnd.ollama.image.model"

// ollamaManifest matches Ollama's on-disk manifest JSON (subset).
type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

func listOllamaLibraryEntries(ollamaModelsDir string) ([]Entry, error) {
	manifestRoot := filepath.Join(ollamaModelsDir, "manifests")
	st, err := os.Stat(manifestRoot)
	if err != nil || !st.IsDir() {
		return nil, nil
	}

	var out []Entry
	err = filepath.WalkDir(manifestRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(manifestRoot, path)
		if err != nil {
			return err
		}
		parts := strings.Split(rel, string(filepath.Separator))
		// Ollama layout: manifests/<host>/<namespace>/<model>/<tag>
		if len(parts) != 4 {
			return nil
		}
		_, _, model, tag := parts[0], parts[1], parts[2], parts[3]

		b, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		var mf ollamaManifest
		if err := json.Unmarshal(b, &mf); err != nil {
			return nil
		}
		var best string
		var bestSize int64
		for _, L := range mf.Layers {
			if L.MediaType == ollamaMediaTypeModel && L.Digest != "" && L.Size >= bestSize {
				bestSize = L.Size
				best = L.Digest
			}
		}
		if best == "" {
			return nil
		}
		blobName := strings.ReplaceAll(best, ":", "-")
		blobPath := filepath.Join(ollamaModelsDir, "blobs", blobName)
		st, err := os.Stat(blobPath)
		if err != nil {
			return nil
		}
		name := ollamaEntryName(model, tag)
		out = append(out, Entry{
			Name:       name,
			Path:       blobPath,
			Size:       st.Size(),
			ModifiedAt: st.ModTime().UTC(),
			Format:     "gguf",
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return dedupeEntriesByName(out), nil
}

func ollamaEntryName(model, tag string) string {
	if tag == "" || strings.EqualFold(tag, "latest") {
		return model
	}
	return model + "-" + tag
}

func dedupeEntriesByName(in []Entry) []Entry {
	seen := make(map[string]Entry)
	for _, e := range in {
		key := strings.ToLower(e.Name)
		if old, ok := seen[key]; ok {
			if e.ModifiedAt.After(old.ModifiedAt) {
				seen[key] = e
			}
			continue
		}
		seen[key] = e
	}
	out := make([]Entry, 0, len(seen))
	for _, e := range seen {
		out = append(out, e)
	}
	return out
}
