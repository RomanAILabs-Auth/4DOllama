package convert

import (
	"fmt"
)

// GGML type ids (ggml.h) — subset + common quants for GGUF conversion.
const (
	GGMLTypeF32  uint32 = 0
	GGMLTypeF16  uint32 = 1
	GGMLTypeQ4_0 uint32 = 2
	GGMLTypeQ4_1 uint32 = 3
	GGMLTypeQ5_0 uint32 = 6
	GGMLTypeQ5_1 uint32 = 7
	GGMLTypeQ8_0 uint32 = 8
	GGMLTypeQ8_1 uint32 = 9
	GGMLTypeQ2_K uint32 = 10
	GGMLTypeQ3_K uint32 = 11
	GGMLTypeQ4_K uint32 = 12
	GGMLTypeQ5_K uint32 = 13
	GGMLTypeQ6_K uint32 = 14
	GGMLTypeQ8_K uint32 = 15
	GGMLTypeBF16 uint32 = 30
)

// ggmlRowSize returns tensor payload size in bytes: type_size * ne / blck_size (llama.cpp ggml_row_size).
func ggmlRowSize(typ uint32, ne int64) (int64, error) {
	if ne < 0 {
		return 0, fmt.Errorf("negative element count")
	}
	bl, ts, ok := ggmlTypeTraits(typ)
	if !ok {
		return 0, fmt.Errorf("unsupported GGML tensor type id %d", typ)
	}
	if bl == 0 || ts == 0 {
		return 0, fmt.Errorf("invalid traits for type %d", typ)
	}
	if ne%bl != 0 {
		return 0, fmt.Errorf("element count %d not divisible by block size %d (type %d)", ne, bl, typ)
	}
	nbl := ne / bl
	return nbl * ts, nil
}

func ggmlTypeTraits(typ uint32) (blckSize int64, typeSize int64, ok bool) {
	switch typ {
	case GGMLTypeF32:
		return 1, 4, true
	case GGMLTypeF16, GGMLTypeBF16:
		return 1, 2, true
	case GGMLTypeQ4_0:
		return 32, 18, true // sizeof(block_q4_0)
	case GGMLTypeQ4_1:
		return 32, 20, true
	case GGMLTypeQ5_0:
		return 32, 22, true
	case GGMLTypeQ5_1:
		return 32, 24, true
	case GGMLTypeQ8_0:
		return 32, 34, true // ggml_half + 32 int8
	case GGMLTypeQ8_1:
		return 32, 36, true
	case GGMLTypeQ2_K:
		return 256, 2*2 + 256/16 + 256/4, true // block_q2_K
	case GGMLTypeQ3_K:
		return 256, 2 + 256/4 + 256/8 + 12, true // block_q3_K
	case GGMLTypeQ4_K:
		return 256, 144, true // 2*ggml_half + 12 + 128
	case GGMLTypeQ5_K:
		return 256, 160, true // 2*2 + 12 + 128 + 32 qh
	case GGMLTypeQ6_K:
		return 256, 210, true // ggml_half + 16 + 3*64
	case GGMLTypeQ8_K:
		return 256, 4 + 256 + (256/16)*2, true // block_q8_K
	default:
		return 0, 0, false
	}
}

func GGMLTypeName(typ uint32) string {
	switch typ {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeBF16:
		return "BF16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ5_1:
		return "Q5_1"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ8_1:
		return "Q8_1"
	case GGMLTypeQ2_K:
		return "Q2_K"
	case GGMLTypeQ3_K:
		return "Q3_K"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeQ8_K:
		return "Q8_K"
	default:
		return fmt.Sprintf("type_%d", typ)
	}
}
