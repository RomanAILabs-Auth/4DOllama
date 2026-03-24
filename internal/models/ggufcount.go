package models

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// CountGGUFParams returns the sum of tensor element counts (metadata scan only).
func CountGGUFParams(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return 0, err
	}
	if string(magic[:]) != "GGUF" {
		return 0, errors.New("not a GGUF file")
	}
	var ver uint32
	if err := binary.Read(f, binary.LittleEndian, &ver); err != nil {
		return 0, err
	}
	if ver < 2 || ver > 4 {
		return 0, fmt.Errorf("unsupported gguf version %d", ver)
	}
	var tensorCount, kvCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return 0, err
	}
	if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return 0, err
	}
	for i := uint64(0); i < kvCount; i++ {
		if _, err := readGGUFString(f); err != nil {
			return 0, err
		}
		var ty uint32
		if err := binary.Read(f, binary.LittleEndian, &ty); err != nil {
			return 0, err
		}
		if err := skipGGUFValue(f, ty); err != nil {
			return 0, err
		}
	}
	var total int64
	for i := uint64(0); i < tensorCount; i++ {
		if _, err := readGGUFString(f); err != nil {
			return 0, err
		}
		var nDims uint32
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			return 0, err
		}
		var ne uint64 = 1
		for d := uint32(0); d < nDims; d++ {
			var dim uint64
			if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
				return 0, err
			}
			ne *= dim
		}
		var tensorMeta struct {
			GgmlType uint32
			Offset   uint64
		}
		if err := binary.Read(f, binary.LittleEndian, &tensorMeta.GgmlType); err != nil {
			return 0, err
		}
		if err := binary.Read(f, binary.LittleEndian, &tensorMeta.Offset); err != nil {
			return 0, err
		}
		total += int64(ne)
	}
	return total, nil
}

func readGGUFString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<30 {
		return "", errors.New("gguf string too long")
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func skipGGUFValue(r io.Reader, ty uint32) error {
	switch ty {
	case 0:
		var x uint8
		return binary.Read(r, binary.LittleEndian, &x)
	case 1:
		var x int8
		return binary.Read(r, binary.LittleEndian, &x)
	case 2:
		var x uint16
		return binary.Read(r, binary.LittleEndian, &x)
	case 3:
		var x int16
		return binary.Read(r, binary.LittleEndian, &x)
	case 4:
		var x uint32
		return binary.Read(r, binary.LittleEndian, &x)
	case 5:
		var x int32
		return binary.Read(r, binary.LittleEndian, &x)
	case 6:
		var x float32
		return binary.Read(r, binary.LittleEndian, &x)
	case 7:
		var x uint8
		return binary.Read(r, binary.LittleEndian, &x)
	case 8:
		_, err := readGGUFString(r)
		return err
	case 9:
		var et uint32
		if err := binary.Read(r, binary.LittleEndian, &et); err != nil {
			return err
		}
		var ne uint64
		if err := binary.Read(r, binary.LittleEndian, &ne); err != nil {
			return err
		}
		for i := uint64(0); i < ne; i++ {
			if err := skipGGUFValue(r, et); err != nil {
				return err
			}
		}
		return nil
	case 10:
		var x uint64
		return binary.Read(r, binary.LittleEndian, &x)
	case 11:
		var x int64
		return binary.Read(r, binary.LittleEndian, &x)
	case 12:
		var x float64
		return binary.Read(r, binary.LittleEndian, &x)
	default:
		return fmt.Errorf("unknown gguf value type %d", ty)
	}
}
