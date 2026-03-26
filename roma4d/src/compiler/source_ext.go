package compiler

import (
	"os"
	"path/filepath"
	"strings"
)

// Roma4DSourceExtR4D is the official Roma4D source suffix.
const Roma4DSourceExtR4D = ".r4d"

// Roma4DSourceExtR4S is a short legacy suffix; still accepted.
const Roma4DSourceExtR4S = ".r4s"

// Roma4DSourceExtLegacy is the original long extension; still accepted.
const Roma4DSourceExtLegacy = ".roma4d"

// Roma4DSourceExtensions lists accepted suffixes, preferred first.
var Roma4DSourceExtensions = []string{Roma4DSourceExtR4D, Roma4DSourceExtR4S, Roma4DSourceExtLegacy}

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

// StripRoma4DSourceExt removes .r4d, .r4s, or .roma4d from a filename (for module naming).
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
// import path, or "" if none found. Tries .r4d, then .r4s, then .roma4d.
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
