package ai

import (
	"testing"
	"unsafe"
)

func TestAlignedFloat64_16ByteAligned(t *testing.T) {
	s := AlignedFloat64(64)
	if len(s) != 64 {
		t.Fatalf("len %d", len(s))
	}
	p := uintptr(unsafe.Pointer(&s[0]))
	if p%16 != 0 {
		t.Fatalf("base %x not 16-byte aligned", p)
	}
}

func TestNewSoA4xN_BasePointer(t *testing.T) {
	s := NewSoA4xN(8)
	if s.N != 8 {
		t.Fatal(s.N)
	}
	bp := s.BasePointer()
	if bp == 0 {
		t.Fatal("nil base")
	}
	if bp%16 != 0 {
		t.Fatalf("SoA base %x not 16-byte aligned", bp)
	}
}

func TestMapPythonBinOpToGA(t *testing.T) {
	if _, ok := MapPythonBinOpToGA("@"); !ok {
		t.Fatal()
	}
}
