package models

import (
	"os"
	"path/filepath"
	"strings"
)

// ReadTokenizerFromModelfile returns the path from a TOKENIZER_FROM line, or "" if missing/unreadable.
func ReadTokenizerFromModelfile(modelfilePath string) string {
	b, err := os.ReadFile(modelfilePath)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		upper := strings.ToUpper(line)
		const prefix = "TOKENIZER_FROM "
		if strings.HasPrefix(upper, prefix) {
			p := strings.TrimSpace(line[len(prefix):])
			p = strings.Trim(p, `"'`)
			if p == "" {
				return ""
			}
			if filepath.IsAbs(p) {
				return filepath.Clean(p)
			}
			dir := filepath.Dir(modelfilePath)
			return filepath.Clean(filepath.Join(dir, p))
		}
	}
	return ""
}

// SiblingModelfilePath returns models/<stem>.Modelfile for a weights file models/<stem>.4dai.
func SiblingModelfilePath(fourDaiPath string) string {
	dir := filepath.Dir(fourDaiPath)
	base := strings.TrimSuffix(filepath.Base(fourDaiPath), filepath.Ext(fourDaiPath))
	return filepath.Join(dir, base+".Modelfile")
}
