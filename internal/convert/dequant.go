package convert

import (
	"encoding/binary"
	"fmt"
	"math"
)

const qkK = 256

// DequantizeGGML turns raw GGUF tensor bytes into float32 element values (one tensor).
func DequantizeGGML(raw []byte, typ uint32, ne int64) ([]float32, error) {
	if ne == 0 {
		return nil, nil
	}
	want, err := ggmlRowSize(typ, ne)
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) != want {
		return nil, fmt.Errorf("tensor byte size mismatch: got %d want %d (type %s ne=%d)", len(raw), want, GGMLTypeName(typ), ne)
	}
	out := make([]float32, ne)
	switch typ {
	case GGMLTypeF32:
		for i := int64(0); i < ne; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case GGMLTypeF16:
		for i := int64(0); i < ne; i++ {
			out[i] = fp16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case GGMLTypeBF16:
		for i := int64(0); i < ne; i++ {
			out[i] = bf16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case GGMLTypeQ4_0:
		dequantQ4_0(raw, out)
		return out, nil
	case GGMLTypeQ4_1:
		dequantQ4_1(raw, out)
		return out, nil
	case GGMLTypeQ5_0:
		dequantQ5_0(raw, out)
		return out, nil
	case GGMLTypeQ5_1:
		dequantQ5_1(raw, out)
		return out, nil
	case GGMLTypeQ8_0:
		dequantQ8_0(raw, out)
		return out, nil
	case GGMLTypeQ4_K:
		dequantQ4_K(raw, out)
		return out, nil
	case GGMLTypeQ6_K:
		dequantQ6_K(raw, out)
		return out, nil
	default:
		return nil, fmt.Errorf("dequant not implemented for %s (id %d) — try an F16 GGUF or extend internal/convert", GGMLTypeName(typ), typ)
	}
}

func dequantQ4_0(raw []byte, y []float32) {
	const qk = 32
	nb := len(raw) / 18
	for i := 0; i < nb; i++ {
		off := i * 18
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		base := i * qk
		for j := 0; j < qk/2; j++ {
			v := raw[off+2+j]
			x0 := float32(int(v&0x0f) - 8)
			x1 := float32(int(v>>4) - 8)
			y[base+j+0] = x0 * d
			y[base+j+16] = x1 * d
		}
	}
}

func dequantQ4_1(raw []byte, y []float32) {
	const qk = 32
	nb := len(raw) / 20
	for i := 0; i < nb; i++ {
		off := i * 20
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		m := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off+2:]))
		base := i * qk
		for j := 0; j < qk/2; j++ {
			v := raw[off+4+j]
			x0 := float32(v & 0x0f)
			x1 := float32(v >> 4)
			y[base+j+0] = x0*d + m
			y[base+j+16] = x1*d + m
		}
	}
}

func dequantQ5_0(raw []byte, y []float32) {
	const qk = 32
	nb := len(raw) / 22
	for i := 0; i < nb; i++ {
		off := i * 22
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		qh := binary.LittleEndian.Uint32(raw[off+2 : off+6])
		base := i * qk
		for j := 0; j < qk/2; j++ {
			xh0 := uint32((qh >> uint(j+0)) << 4) & 0x10
			xh1 := uint32((qh >> uint(j+12))) & 0x10
			v := raw[off+6+j]
			x0 := float32(int32((uint32(v&0x0f) | xh0) - 16))
			x1 := float32(int32((uint32(v>>4) | xh1) - 16))
			y[base+j+0] = x0 * d
			y[base+j+16] = x1 * d
		}
	}
}

func dequantQ5_1(raw []byte, y []float32) {
	const qk = 32
	nb := len(raw) / 24
	for i := 0; i < nb; i++ {
		off := i * 24
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		m := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off+2:]))
		qh := binary.LittleEndian.Uint32(raw[off+4 : off+8])
		base := i * qk
		for j := 0; j < qk/2; j++ {
			xh0 := uint32((qh >> uint(j+0)) << 4) & 0x10
			xh1 := uint32((qh >> uint(j+12))) & 0x10
			v := raw[off+8+j]
			x0 := float32(int32((uint32(v&0x0f) | xh0)))
			x1 := float32(int32((uint32(v>>4) | xh1)))
			y[base+j+0] = x0*d + m
			y[base+j+16] = x1*d + m
		}
	}
}

func dequantQ8_0(raw []byte, y []float32) {
	const qk = 32
	nb := len(raw) / 34
	for i := 0; i < nb; i++ {
		off := i * 34
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		base := i * qk
		for j := 0; j < qk; j++ {
			y[base+j] = float32(int8(raw[off+2+j])) * d
		}
	}
}

func getScaleMinK4(j int, q []byte) (sc, mn uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
}

// block_q6_K layout (ggml-common.h): ql[128], qh[64], scales[16] int8, d fp16 — 210 bytes / 256 weights.
func dequantQ6_K(raw []byte, y []float32) {
	const qk = 256
	const block = 210
	nb := len(raw) / block
	yi := 0
	for i := 0; i < nb; i++ {
		off := i * block
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off+208 : off+210]))
		ql := raw[off : off+128]
		qh := raw[off+128 : off+192]
		scBytes := raw[off+192 : off+208]
		yy := y[yi : yi+qk]
		yi += qk

		qlOff, qhOff, scOff := 0, 0, 0
		yp := 0
		for range 2 { // two 128-element stripes (matches llama.cpp pointer advance)
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int32(int8((ql[qlOff+l]&0x0f)|((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int32(int8((ql[qlOff+l+32]&0x0f)|((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int32(int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int32(int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4)) - 32
				s0 := float32(int8(scBytes[scOff+is+0]))
				s2 := float32(int8(scBytes[scOff+is+2]))
				s4 := float32(int8(scBytes[scOff+is+4]))
				s6 := float32(int8(scBytes[scOff+is+6]))
				yy[yp+l+0] = d * s0 * float32(q1)
				yy[yp+l+32] = d * s2 * float32(q2)
				yy[yp+l+64] = d * s4 * float32(q3)
				yy[yp+l+96] = d * s6 * float32(q4)
			}
			yp += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
}

func dequantQ4_K(raw []byte, y []float32) {
	nb := len(raw) / 144
	yi := 0
	for i := 0; i < nb; i++ {
		off := i * 144
		d := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		min := fp16ToFloat32(binary.LittleEndian.Uint16(raw[off+2:]))
		scales := raw[off+4 : off+16]
		qb := raw[off+16 : off+144]
		is := 0
		q := 0
		for j := 0; j < qkK; j += 64 {
			sc, m := getScaleMinK4(is+0, scales)
			d1 := d * float32(sc)
			m1 := min * float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2 := d * float32(sc)
			m2 := min * float32(m)
			for l := 0; l < 32; l++ {
				y[yi+l] = d1*float32(qb[q+l]&0x0f) - m1
			}
			for l := 0; l < 32; l++ {
				y[yi+32+l] = d2*float32(qb[q+l]>>4) - m2
			}
			q += 32
			is += 2
		}
		yi += qkK
	}
}
