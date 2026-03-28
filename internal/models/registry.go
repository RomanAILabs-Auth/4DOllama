package models

import (
	"encoding/binary"
	"errors"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Entry describes a discovered model file on disk (GGUF, romanai .4dai, or Ollama blob).
type Entry struct {
	Name       string
	Path       string
	Size       int64
	ModifiedAt time.Time
	// Format is "gguf" or "4dai" (JSON romanai.4dai or ROMANAI4+safetensors shard).
	Format string
	// ShardPaths is non-empty for multi-part native .4dai (see .multi4dai manifest); Path is the first shard.
	ShardPaths []string
	// MultiManifestPath is set when the model is registered via <name>.multi4dai (Path points at a blob, not this file).
	MultiManifestPath string
	// TokenizerGGUF is an optional GGUF path (metadata-only read) for tokenizer.ggml.tokens when weights are .4dai.
	TokenizerGGUF string
}

// Registry scans FOURD_MODELS for *.gguf and optionally merges Ollama's manifests + shared blobs.
type Registry struct {
	dir         string
	ollamaRoot  string
	shareOllama bool
	log         *slog.Logger
}

// NewRegistry builds a registry. When shareOllama is true, models pulled into Ollama appear here without copying.
func NewRegistry(modelsDir, ollamaModelsDir string, shareOllama bool, log *slog.Logger) *Registry {
	return &Registry{
		dir:         modelsDir,
		ollamaRoot:  ollamaModelsDir,
		shareOllama: shareOllama,
		log:         log,
	}
}

// Dir returns the configured 4dollama models directory (extra GGUFs / hardlinks).
func (r *Registry) Dir() string {
	return r.dir
}

// OllamaDir returns the Ollama models root (manifests + blobs).
func (r *Registry) OllamaDir() string {
	return r.ollamaRoot
}

func (r *Registry) listGGUF() ([]Entry, error) {
	_ = os.MkdirAll(r.dir, 0o755)
	var out []Entry
	err := filepath.WalkDir(r.dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !strings.EqualFold(filepath.Ext(path), ".gguf") {
			return nil
		}
		st, err := d.Info()
		if err != nil {
			return err
		}
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		out = append(out, Entry{
			Name:       name,
			Path:       path,
			Size:       st.Size(),
			ModifiedAt: st.ModTime().UTC(),
			Format:     "gguf",
		})
		return nil
	})
	return out, err
}

func (r *Registry) list4DAI() ([]Entry, error) {
	_ = os.MkdirAll(r.dir, 0o755)
	var out []Entry
	err := filepath.WalkDir(r.dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, rerr := filepath.Rel(r.dir, path)
		if rerr == nil {
			parts := strings.Split(rel, string(filepath.Separator))
			if len(parts) > 0 && parts[0] == "blobs" {
				return nil
			}
		}
		if !strings.EqualFold(filepath.Ext(path), ".4dai") {
			return nil
		}
		st, err := d.Info()
		if err != nil {
			return err
		}
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		e := Entry{
			Name:       name,
			Path:       path,
			Size:       st.Size(),
			ModifiedAt: st.ModTime().UTC(),
			Format:     "4dai",
		}
		if tok := ReadTokenizerFromModelfile(SiblingModelfilePath(path)); tok != "" {
			e.TokenizerGGUF = tok
		}
		out = append(out, e)
		return nil
	})
	return out, err
}

// List returns models from ~/.4dollama/models/*.gguf plus, when sharing is on, Ollama library manifests.
// Local .gguf files override same logical name as Ollama entries.
func (r *Registry) List() ([]Entry, error) {
	byName := make(map[string]Entry)

	if r.shareOllama && r.ollamaRoot != "" {
		oll, err := listOllamaLibraryEntries(r.ollamaRoot)
		if err != nil && r.log != nil {
			r.log.Debug("ollama manifest scan", slog.Any("err", err))
		}
		for _, e := range oll {
			byName[strings.ToLower(e.Name)] = e
		}
	}

	local, err := r.listGGUF()
	if err != nil {
		return nil, err
	}
	for _, e := range local {
		byName[strings.ToLower(e.Name)] = e
	}

	dai, err := r.list4DAI()
	if err != nil {
		return nil, err
	}
	for _, e := range dai {
		k := strings.ToLower(e.Name)
		if _, ok := byName[k]; ok {
			alt := e
			alt.Name = e.Name + "-romanai"
			byName[strings.ToLower(alt.Name)] = alt
		} else {
			byName[k] = e
		}
	}

	sharded, err := ListRomanaiShardedEntries(r.dir)
	if err != nil && r.log != nil {
		r.log.Debug("sharded romanai list", slog.Any("err", err))
	}
	for _, e := range sharded {
		byName[strings.ToLower(e.Name)] = e
	}

	out := make([]Entry, 0, len(byName))
	for _, e := range byName {
		out = append(out, e)
	}
	return out, nil
}

// Resolve finds a model by name (stem) or returns ("", false).
func (r *Registry) Resolve(name string) (Entry, bool) {
	name = strings.TrimSpace(name)
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[:i]
	}
	key := strings.ToLower(name)

	// Prefer local .gguf then local .4dai
	local, err := r.listGGUF()
	if err == nil {
		for _, e := range local {
			if strings.ToLower(e.Name) == key {
				return e, true
			}
		}
	} else if r.log != nil {
		r.log.Warn("local models list failed", slog.Any("err", err))
	}
	dai, err := r.list4DAI()
	if err == nil {
		for _, e := range dai {
			if strings.ToLower(e.Name) == key {
				return e, true
			}
		}
	}
	if e, ok := FindShardedRomanaiEntry(r.dir, key); ok {
		return e, true
	}
	if strings.HasSuffix(key, "-romanai") {
		stem := strings.TrimSuffix(key, "-romanai")
		if dai2, err := r.list4DAI(); err == nil {
			for _, e := range dai2 {
				if strings.ToLower(e.Name) == stem {
					return e, true
				}
			}
		}
	}

	if r.shareOllama && r.ollamaRoot != "" {
		oll, err := listOllamaLibraryEntries(r.ollamaRoot)
		if err == nil {
			for _, e := range oll {
				if strings.ToLower(e.Name) == key {
					return e, true
				}
			}
			// e.g. user says "qwen2.5" but only "qwen2.5-7b" exists
			var hits []Entry
			prefix := key + "-"
			for _, e := range oll {
				en := strings.ToLower(e.Name)
				if strings.HasPrefix(en, prefix) {
					hits = append(hits, e)
				}
			}
			if len(hits) == 1 {
				return hits[0], true
			}
		}
	}

	return Entry{}, false
}

// ModelStem4DGGUF normalizes a model ref (e.g. "qwen2.5:latest" → "qwen2.5") for .4dgguf filenames.
func ModelStem4DGGUF(modelRef string) string {
	s := strings.TrimSpace(modelRef)
	if i := strings.IndexByte(s, ':'); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	s = strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == 0 {
			return '_'
		}
		return r
	}, s)
	if s == "" {
		return "model"
	}
	return s
}

// BlobPath4DGGUF returns ~/.ollama/models/blobs/<stem>.4dgguf when ollamaModelsRoot is the Ollama models dir.
func BlobPath4DGGUF(ollamaModelsRoot, modelRef string) string {
	stem := ModelStem4DGGUF(modelRef)
	return filepath.Join(ollamaModelsRoot, "blobs", stem+".4dgguf")
}

var magic4DGGUF = []byte{'4', 'D', 'G', 'F'}

// Save4DGGUF writes lifted 4D float weights + header (native tensor cache).
func Save4DGGUF(path string, weights []float32, paramCount int64) error {
	if len(weights) == 0 {
		return nil
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(magic4DGGUF); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, paramCount); err != nil {
		return err
	}
	nf := uint32(len(weights))
	if err := binary.Write(f, binary.LittleEndian, nf); err != nil {
		return err
	}
	for _, v := range weights {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

// Load4DGGUF reads a .4dgguf blob; returns weights and recorded param count.
func Load4DGGUF(path string) ([]float32, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	var mag [4]byte
	if _, err := f.Read(mag[:]); err != nil {
		return nil, 0, err
	}
	if string(mag[:]) != string(magic4DGGUF) {
		return nil, 0, errors.New("4dgguf: bad magic")
	}
	var ver uint32
	if err := binary.Read(f, binary.LittleEndian, &ver); err != nil {
		return nil, 0, err
	}
	if ver != 1 {
		return nil, 0, errors.New("4dgguf: unsupported version")
	}
	var paramCount int64
	if err := binary.Read(f, binary.LittleEndian, &paramCount); err != nil {
		return nil, 0, err
	}
	var nf uint32
	if err := binary.Read(f, binary.LittleEndian, &nf); err != nil {
		return nil, 0, err
	}
	if nf == 0 || nf > 1<<28 {
		return nil, 0, errors.New("4dgguf: bad float count")
	}
	out := make([]float32, nf)
	for i := range out {
		if err := binary.Read(f, binary.LittleEndian, &out[i]); err != nil {
			return nil, 0, err
		}
	}
	return out, paramCount, nil
}
