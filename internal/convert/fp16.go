package convert

import (
	"encoding/binary"
	"math"
)

// fp16ToFloat32 decodes IEEE 754 binary16 (ggml_half).
func fp16ToFloat32(h uint16) float32 {
	s := uint32(h>>15) << 31
	e := (h >> 10) & 0x1f
	m := uint32(h & 0x3ff)
	switch e {
	case 0:
		if m == 0 {
			return math.Float32frombits(s)
		}
		for m&0x400 == 0 {
			m <<= 1
			e--
		}
		e++
		m &= 0x3ff
	case 31:
		return math.Float32frombits(s | 0x7f800000 | (m << 13))
	}
	e += 127 - 15
	return math.Float32frombits(s | (uint32(e) << 23) | (m << 13))
}

func bf16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

func float32ToFp16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	s := (b >> 16) & 0x8000
	e := int((b >> 23)&0xff) - 127 + 15
	m := b & 0x007fffff
	if e <= 0 {
		if e < -10 {
			return uint16(s)
		}
		m |= 0x00800000
		shift := uint(1 - e)
		if shift > 23 {
			return uint16(s)
		}
		m >>= shift
		return uint16(s | uint32(m))
	}
	if e >= 31 {
		if f > 0 {
			return uint16(s | 0x7c00)
		}
		return uint16(s | 0x7c00)
	}
	return uint16(s | uint32(e)<<10 | (m >> 13))
}

func appendF16LE(dst []byte, f float32) []byte {
	u := float32ToFp16Bits(f)
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], u)
	return append(dst, b[:]...)
}
