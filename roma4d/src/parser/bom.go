package parser

import "strings"

// stripLeadingUTF8BOM removes one or more leading U+FEFF runes (UTF-8 BOM).
// Editors and PowerShell often emit BOM; combining UTF8Encoding(true) with a literal
// U+FEFF in the string can produce a double BOM — TrimLeft handles that.
func stripLeadingUTF8BOM(s string) string {
	return strings.TrimLeft(s, "\ufeff")
}
