package compiler

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// IncrementalCacheDir returns the default cache directory under the package root (Pass 9).
func IncrementalCacheDir(pkgRoot string) string {
	return filepath.Join(pkgRoot, ".roma4d", "cache")
}

// PackageInputHash fingerprints roma4d.toml and a single entry source file for
// dependency-style incremental keys. Deeper import graphs can extend this later.
func PackageInputHash(pkgRoot, entryPath string) (string, error) {
	h := sha256.New()
	tomlPath := filepath.Join(pkgRoot, "roma4d.toml")
	b, err := os.ReadFile(tomlPath)
	if err != nil {
		return "", fmt.Errorf("incremental: read roma4d.toml: %w", err)
	}
	if _, err := h.Write(b); err != nil {
		return "", err
	}
	entry := entryPath
	if !filepath.IsAbs(entry) {
		entry = filepath.Join(pkgRoot, entryPath)
	}
	f, err := os.Open(entry)
	if err != nil {
		return "", fmt.Errorf("incremental: open entry %s: %w", entry, err)
	}
	defer f.Close()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
