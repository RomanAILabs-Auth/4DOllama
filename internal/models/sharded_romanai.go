package models

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	romanaiShardedFormat = "romanai.sharded.v1"
	romanaiMultiExt      = ".multi4dai"
)

// RomanaiShardedManifest lists blob basenames (under models/blobs/) for a single logical model.
type RomanaiShardedManifest struct {
	Format string   `json:"format"`
	Name   string   `json:"name"`
	Blobs  []string `json:"blobs"`
}

// LooksLikeSafetensors4DAI returns true if the file starts with a plausible safetensors header length + JSON.
func LooksLikeSafetensors4DAI(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var lenBuf [8]byte
	if _, err := f.Read(lenBuf[:]); err != nil {
		return false
	}
	n := binary.LittleEndian.Uint64(lenBuf[:])
	if n == 0 || n > 256<<20 {
		return false
	}
	hdr := make([]byte, n)
	if _, err := f.Read(hdr); err != nil {
		return false
	}
	i := 0
	for i < len(hdr) && (hdr[i] == ' ' || hdr[i] == '\n' || hdr[i] == '\t' || hdr[i] == '\r') {
		i++
	}
	return i < len(hdr) && hdr[i] == '{'
}

// IsNative4DAIWeight returns true for ROMANAI4 envelope or safetensors-style .4dai shards (not JSON text).
func IsNative4DAIWeight(path string) (bool, error) {
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".4dai" {
		return false, nil
	}
	ok, err := IsRomanaiBinary4DAI(path)
	if err != nil || ok {
		return ok, err
	}
	return LooksLikeSafetensors4DAI(path), nil
}

// ExpandRomanaiV2PartShards expands FROM ./foo_part1.4dai into part1..part4 when all exist.
// If only part1 exists, returns a single-element slice. If part1 and part2 exist but not 3 or 4, returns an error.
func ExpandRomanaiV2PartShards(absFirst string) ([]string, error) {
	absFirst = filepath.Clean(absFirst)
	dir, stem, ok := parseRomanaiV2Part1(absFirst)
	if !ok {
		return []string{absFirst}, nil
	}
	var paths []string
	for part := 1; part <= 4; part++ {
		cand := filepath.Join(dir, fmt.Sprintf("%s_part%d.4dai", stem, part))
		if _, err := os.Stat(cand); err != nil {
			if part == 1 {
				return nil, fmt.Errorf("romanai shard: %w", err)
			}
			break
		}
		paths = append(paths, cand)
	}
	if len(paths) > 1 && len(paths) != 4 {
		return nil, fmt.Errorf("romanai v2 multi-shard: expected _part1.._part4 (4 files) in %s, found %d", dir, len(paths))
	}
	return paths, nil
}

func parseRomanaiV2Part1(absPath string) (dir, stem string, ok bool) {
	base := filepath.Base(absPath)
	if !strings.EqualFold(filepath.Ext(base), ".4dai") {
		return "", "", false
	}
	name := strings.TrimSuffix(base, filepath.Ext(base))
	low := strings.ToLower(name)
	suf := "_part1"
	if !strings.HasSuffix(low, suf) {
		return "", "", false
	}
	stem = name[:len(name)-len(suf)]
	if stem == "" {
		return "", "", false
	}
	return filepath.Dir(absPath), stem, true
}

// WriteRomanaiShardedManifest writes models/<name>.multi4dai.
func WriteRomanaiShardedManifest(modelsDir, modelName string, blobBasenames []string) error {
	if modelName == "" || len(blobBasenames) == 0 {
		return errors.New("sharded manifest: empty name or blobs")
	}
	m := RomanaiShardedManifest{
		Format: romanaiShardedFormat,
		Name:   modelName,
		Blobs:  append([]string(nil), blobBasenames...),
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	_ = os.MkdirAll(modelsDir, 0o755)
	path := filepath.Join(modelsDir, modelName+romanaiMultiExt)
	return os.WriteFile(path, b, 0o644)
}

// ReadRomanaiShardedManifest reads a .multi4dai JSON file.
func ReadRomanaiShardedManifest(path string) (RomanaiShardedManifest, error) {
	var m RomanaiShardedManifest
	b, err := os.ReadFile(path)
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, err
	}
	if m.Format != romanaiShardedFormat {
		return m, fmt.Errorf("unknown sharded format %q", m.Format)
	}
	if m.Name == "" || len(m.Blobs) == 0 {
		return m, errors.New("invalid sharded manifest")
	}
	return m, nil
}

func shardedBlobAbs(modelsDir, base string) string {
	return filepath.Join(modelsDir, "blobs", base)
}

// EntryFromShardedManifest builds a registry entry (aggregate size, shard paths).
func EntryFromShardedManifest(modelsDir, manifestPath string, m RomanaiShardedManifest) (Entry, error) {
	var paths []string
	var total int64
	var latest time.Time
	for _, b := range m.Blobs {
		if b == "" || strings.Contains(b, string(filepath.Separator)) || strings.Contains(b, "/") || strings.Contains(b, `\`) {
			return Entry{}, fmt.Errorf("invalid blob name %q", b)
		}
		abs := shardedBlobAbs(modelsDir, b)
		st, err := os.Stat(abs)
		if err != nil {
			return Entry{}, fmt.Errorf("blob %s: %w", abs, err)
		}
		paths = append(paths, abs)
		total += st.Size()
		if st.ModTime().After(latest) {
			latest = st.ModTime()
		}
	}
	if len(paths) == 0 {
		return Entry{}, errors.New("no shards")
	}
	e := Entry{
		Name:              m.Name,
		Path:              paths[0],
		Size:              total,
		ModifiedAt:        latest.UTC(),
		Format:            "4dai",
		ShardPaths:        paths,
		MultiManifestPath: manifestPath,
	}
	if tok := ReadTokenizerFromModelfile(filepath.Join(modelsDir, m.Name+".Modelfile")); tok != "" {
		e.TokenizerGGUF = tok
	}
	return e, nil
}

// FindShardedRomanaiEntry resolves a model by manifest Name (case-insensitive) regardless of .multi4dai filename casing.
func FindShardedRomanaiEntry(modelsDir string, keyLower string) (Entry, bool) {
	matches, err := filepath.Glob(filepath.Join(modelsDir, "*"+romanaiMultiExt))
	if err != nil {
		return Entry{}, false
	}
	for _, man := range matches {
		m, err := ReadRomanaiShardedManifest(man)
		if err != nil {
			continue
		}
		if strings.ToLower(m.Name) != keyLower {
			continue
		}
		e, err := EntryFromShardedManifest(modelsDir, man, m)
		if err != nil {
			continue
		}
		return e, true
	}
	return Entry{}, false
}

// ListRomanaiShardedEntries returns one Entry per *.multi4dai in modelsDir (non-recursive).
func ListRomanaiShardedEntries(modelsDir string) ([]Entry, error) {
	_ = os.MkdirAll(modelsDir, 0o755)
	matches, err := filepath.Glob(filepath.Join(modelsDir, "*"+romanaiMultiExt))
	if err != nil {
		return nil, err
	}
	var out []Entry
	for _, man := range matches {
		m, err := ReadRomanaiShardedManifest(man)
		if err != nil {
			continue
		}
		e, err := EntryFromShardedManifest(modelsDir, man, m)
		if err != nil {
			continue
		}
		out = append(out, e)
	}
	return out, nil
}

// ParseRomanaiV2PartStem extracts "romanai_v2" from "romanai_v2_part3.4dai" (for tests / tooling).
func ParseRomanaiV2PartStem(filename string) (stem string, part int, ok bool) {
	base := filepath.Base(filename)
	if !strings.EqualFold(filepath.Ext(base), ".4dai") {
		return "", 0, false
	}
	name := strings.TrimSuffix(base, filepath.Ext(base))
	idx := strings.LastIndex(strings.ToLower(name), "_part")
	if idx < 0 {
		return "", 0, false
	}
	n, err := strconv.Atoi(name[idx+len("_part"):])
	if err != nil || n < 1 {
		return "", 0, false
	}
	return name[:idx], n, true
}
