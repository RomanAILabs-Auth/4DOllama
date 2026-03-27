package quantum

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"math"
)

// StateHashSHA256 hashes ReA‖ImA in node-major order (little-endian float64).
func (S *Simulator) StateHashSHA256() [32]byte {
	h := sha256.New()
	n := S.N * S.Dim
	writeFloat64Slice(h, S.ReA[:n])
	writeFloat64Slice(h, S.ImA[:n])
	var out [32]byte
	h.Sum(out[:0])
	return out
}

func writeFloat64Slice(h hash.Hash, buf []float64) {
	var b [8]byte
	for _, v := range buf {
		u := math.Float64bits(v)
		binary.LittleEndian.PutUint64(b[:], u)
		h.Write(b[:])
	}
}
