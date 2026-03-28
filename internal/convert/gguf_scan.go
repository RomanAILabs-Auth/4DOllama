package convert

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// GGUFTensor is one tensor entry from a GGUF v2–v4 file.
type GGUFTensor struct {
	Name     string
	NDims    uint32
	Shape    []uint64 // row-major, GGUF order
	GGMLType uint32
	Offset   uint64 // byte offset from start of tensor data region
	NElems   int64  // product of shape
}

// GGUFLayout holds scan results for conversion.
type GGUFLayout struct {
	Version     uint32
	TensorCount uint64
	KVCount     uint64
	Architecture string
	Tensors     []GGUFTensor
	DataBase    uint64 // absolute file offset where tensor bytes begin (32-byte aligned)
	FileSize    int64
}

// ScanGGUF reads GGUF header and tensor index. Does not read tensor payloads.
func ScanGGUF(path string) (*GGUFLayout, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, err
	}
	if string(magic[:]) != "GGUF" {
		return nil, errors.New("not a GGUF file (expected GGUF magic)")
	}
	var ver uint32
	if err := binary.Read(f, binary.LittleEndian, &ver); err != nil {
		return nil, err
	}
	if ver < 2 || ver > 4 {
		return nil, fmt.Errorf("unsupported GGUF version %d", ver)
	}
	var tensorCount, kvCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return nil, err
	}

	var arch string
	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			return nil, err
		}
		var ty uint32
		if err := binary.Read(f, binary.LittleEndian, &ty); err != nil {
			return nil, err
		}
		if key == "general.architecture" && ty == 8 {
			s, err := readGGUFString(f)
			if err != nil {
				return nil, err
			}
			arch = s
		} else {
			if err := skipGGUFValue(f, ty); err != nil {
				return nil, err
			}
		}
	}

	tensors := make([]GGUFTensor, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readGGUFString(f)
		if err != nil {
			return nil, err
		}
		var nDims uint32
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			return nil, err
		}
		shape := make([]uint64, nDims)
		var ne int64 = 1
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &shape[d]); err != nil {
				return nil, err
			}
			ne *= int64(shape[d])
		}
		var ggmlType uint32
		if err := binary.Read(f, binary.LittleEndian, &ggmlType); err != nil {
			return nil, err
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, err
		}
		tensors = append(tensors, GGUFTensor{
			Name:     name,
			NDims:    nDims,
			Shape:    shape,
			GGMLType: ggmlType,
			Offset:   offset,
			NElems:   ne,
		})
	}

	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	const align = int64(32)
	dataBase := (pos + align - 1) / align * align

	return &GGUFLayout{
		Version:      ver,
		TensorCount:  tensorCount,
		KVCount:      kvCount,
		Architecture: arch,
		Tensors:      tensors,
		DataBase:     uint64(dataBase),
		FileSize:     st.Size(),
	}, nil
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
		var x bool
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
