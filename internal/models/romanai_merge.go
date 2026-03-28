package models

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const romanaiJSONFormat = "romanai.4dai"

// RomanaiJSONEnvelope matches the RQ4D / 4dollama JSON .4dai layout (human-auditable carrier).
type RomanaiJSONEnvelope struct {
	Header map[string]any `json:"header"`
	Layers []any            `json:"layers"`
}

// IsRomanaiBinary4DAI returns true if the file begins with the Phase-3 ROMANAI8 magic (safetensors envelope).
func IsRomanaiBinary4DAI(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	var buf [8]byte
	n, err := f.Read(buf[:])
	if err != nil || n < 8 {
		return false, err
	}
	return string(buf[:]) == "ROMANAI4", nil
}

// MergeRomanaiJSON4DAIFiles merges multiple JSON romanai.4dai shards (concatenates layers, unions metadata).
// Binary .4dai shards must be merged offline (single FROM only); this returns an error if any shard is binary.
func MergeRomanaiJSON4DAIFiles(shardPaths []string, outPath string) error {
	if len(shardPaths) == 0 {
		return errors.New("merge: no shard paths")
	}
	var merged RomanaiJSONEnvelope
	for i, p := range shardPaths {
		p = filepath.Clean(p)
		bin, err := IsRomanaiBinary4DAI(p)
		if err != nil {
			return fmt.Errorf("shard %q: %w", p, err)
		}
		if bin {
			return fmt.Errorf("shard %q: binary ROMANAI4+safetensors — use a single FROM or merge with RomanAILabs ai4d_format.py", p)
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("shard %q: %w", p, err)
		}
		if !json.Valid(bytes.TrimSpace(raw)) {
			return fmt.Errorf("shard %q: not valid JSON", p)
		}
		var env RomanaiJSONEnvelope
		if err := json.Unmarshal(raw, &env); err != nil {
			return fmt.Errorf("shard %q: %w", p, err)
		}
		if env.Header == nil {
			return fmt.Errorf("shard %q: missing header", p)
		}
		fmtVal, _ := env.Header["format"].(string)
		if fmtVal != "" && fmtVal != romanaiJSONFormat {
			return fmt.Errorf("shard %q: expected format %q, got %q", p, romanaiJSONFormat, fmtVal)
		}
		if i == 0 {
			merged.Header = env.Header
			merged.Layers = append([]any(nil), env.Layers...)
		} else {
			merged.Layers = append(merged.Layers, env.Layers...)
		}
	}
	if merged.Header == nil {
		merged.Header = map[string]any{}
	}
	merged.Header["format"] = romanaiJSONFormat
	merged.Header["merged_shard_count"] = float64(len(shardPaths))
	out, err := json.MarshalIndent(merged, "", "  ")
	if err != nil {
		return err
	}
	_ = os.MkdirAll(filepath.Dir(outPath), 0o755)
	return os.WriteFile(outPath, out, 0o644)
}
