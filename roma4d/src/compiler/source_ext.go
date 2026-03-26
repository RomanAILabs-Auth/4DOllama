package compiler

import (
	"os"
	"path/filepath"
	"strings"
)

// Roma4DSourceExtR4S is the preferred source suffix (short, "Roma4D source").
const Roma4DSourceExtR4S = ".r4s"

// Roma4DSourceExtLegacy is the original extension; still accepted everywhere.
const Roma4DSourceExtLegacy = ".roma4d"

// Roma4DSourceExtensions lists accepted suffixes, preferred first.
var Roma4DSourceExtensions = []string{Roma4DSourceExtR4S, Roma4DSourceExtLegacy}

// IsRoma4DSourcePath returns true if path ends with a known Roma4D source extension.
func IsRoma4DSourcePath(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, e := range Roma4DSourceExtensions {
		if ext == strings.ToLower(e) {
			return true
		}
	}
	return false
}

// StripRoma4DSourceExt removes .r4s or .roma4d from a filename (for module naming).
func StripRoma4DSourceExt(name string) string {
	lower := strings.ToLower(name)
	for _, e := range Roma4DSourceExtensions {
		el := strings.ToLower(e)
		if strings.HasSuffix(lower, el) {
			return name[:len(name)-len(e)]
		}
	}
	return name
}

// ResolveRoma4DModuleFile returns the path to an existing module file under root for a dotted
// import path, or "" if none found. Tries .r4s before .roma4d.
func ResolveRoma4DModuleFile(rootDir, dotPath string) string {
	slashPath := strings.ReplaceAll(dotPath, ".", string(filepath.Separator))
	bases := []string{
		filepath.Join(rootDir, slashPath),
		filepath.Join(rootDir, dotPath),
	}
	for _, base := range bases {
		for _, ext := range Roma4DSourceExtensions {
			p := base + ext
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p
			}
		}
	}
	return ""
}
