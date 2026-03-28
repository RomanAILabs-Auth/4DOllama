package convert

import (
	"math"
	"testing"
)

func TestGgmlRowSizeQ4_K(t *testing.T) {
	n, err := ggmlRowSize(GGMLTypeQ4_K, 256)
	if err != nil {
		t.Fatal(err)
	}
	if n != 144 {
		t.Fatalf("got %d want 144", n)
	}
}

func TestGgmlRowSizeF32(t *testing.T) {
	n, err := ggmlRowSize(GGMLTypeF32, 100)
	if err != nil {
		t.Fatal(err)
	}
	if n != 400 {
		t.Fatalf("got %d", n)
	}
}

func TestPackedCliffordF16Bytes(t *testing.T) {
	// 17 elems → 2 blocks of 16 → 32 F16 values → 64 bytes
	if PackedCliffordF16Bytes(17) != 64 {
		t.Fatalf("got %d", PackedCliffordF16Bytes(17))
	}
}

func TestFp16RoundTrip(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 0.5, 3.14159, 1e-3} {
		u := float32ToFp16Bits(v)
		w := fp16ToFloat32(u)
		if math.Abs(float64(v-w)) > 0.01*math.Max(1, math.Abs(float64(v))) && !(v == 0 && w == 0) {
			t.Fatalf("%v -> %v -> %v", v, u, w)
		}
	}
}

func TestDequantQ4_0Smoke(t *testing.T) {
	raw := make([]byte, 18)
	// d=1.0 as fp16 = 0x3c00
	raw[0] = 0x00
	raw[1] = 0x3c
	// qs[0] low nibble 8 -> 0, high 8 -> 0
	raw[2] = 0x88
	y := make([]float32, 32)
	dequantQ4_0(raw, y)
	if math.Abs(float64(y[0])) > 1e-5 {
		t.Fatalf("y[0]=%v", y[0])
	}
}
