package quantum

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"math"
)

// StateHashSHA256 hashes ψ (ReA‖ImA). For BackendTN, appends flattened ρ and bond Schmidt weights.
func (S *Simulator) StateHashSHA256() [32]byte {
	h := sha256.New()
	n := S.N * S.Dim
	writeFloat64Slice(h, S.ReA[:n])
	writeFloat64Slice(h, S.ImA[:n])
	if S.Backend == BackendTN && S.RhoA != nil {
		nr := S.N * S.Dim * S.Dim * 2
		writeFloat64Slice(h, S.RhoA[:nr])
		for ei := range S.Bonds {
			b := &S.Bonds[ei]
			var u8 [4]byte
			binary.LittleEndian.PutUint32(u8[:], uint32(b.Chi))
			h.Write(u8[:])
			for k := 0; k < b.Chi && k < MaxChiCap; k++ {
				writeFloat64(h, b.SingularValues[k])
			}
		}
	}
	var out [32]byte
	h.Sum(out[:0])
	return out
}

func writeFloat64(h hash.Hash, v float64) {
	var b [8]byte
	u := math.Float64bits(v)
	binary.LittleEndian.PutUint64(b[:], u)
	h.Write(b[:])
}

func writeFloat64Slice(h hash.Hash, buf []float64) {
	var b [8]byte
	for _, v := range buf {
		u := math.Float64bits(v)
		binary.LittleEndian.PutUint64(b[:], u)
		h.Write(b[:])
	}
}
