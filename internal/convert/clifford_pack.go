package convert

import (
	"math"
)

// CliffordShape returns [nBlocks, 4, 4] for Cl(4,0) carrier layout (pad tail with zeros).
func CliffordShape(ne int64) []int64 {
	if ne <= 0 {
		return []int64{0, 4, 4}
	}
	nBlocks := (ne + 15) / 16
	return []int64{nBlocks, 4, 4}
}

// PackCliffordF16 flattens float weights into row-major F16 bytes for shape (nBlocks,4,4).
func PackCliffordF16(weights []float32) ([]byte, []int64) {
	ne := int64(len(weights))
	shape := CliffordShape(ne)
	nBlocks := shape[0]
	want := int(nBlocks * 16)
	out := make([]byte, 0, want*2)
	for i := 0; i < want; i++ {
		var f float32
		if i < len(weights) {
			f = weights[i]
		}
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			f = 0
		}
		out = appendF16LE(out, f)
	}
	return out, shape
}
