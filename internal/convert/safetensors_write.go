package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type stTensorDesc struct {
	Dtype       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

// PackedCliffordF16Bytes returns byte length for F16 payload of shape CliffordShape(ne).
func PackedCliffordF16Bytes(ne int64) int64 {
	if ne <= 0 {
		return 0
	}
	n := (ne + 15) / 16 * 16
	return n * 2
}

// WriteSafetensorsCliffordF16Stream writes a .4dai-compatible safetensors file in sorted-key order.
// neByName maps each safetensors key to GGUF element count; produce is called once per tensor.
func WriteSafetensorsCliffordF16Stream(path string, names []string, neByName map[string]int64, produce func(safeName string) ([]byte, []int64, error)) error {
	if len(names) == 0 {
		return fmt.Errorf("no tensors")
	}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)

	header := make(map[string]stTensorDesc, len(sorted))
	var off uint64
	for _, name := range sorted {
		ne := neByName[name]
		n := uint64(PackedCliffordF16Bytes(ne))
		header[name] = stTensorDesc{
			Dtype:       "F16",
			Shape:       CliffordShape(ne),
			DataOffsets: []uint64{off, off + n},
		}
		off += n
	}

	jsonBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}
	var prelude []byte
	prelude = binary.LittleEndian.AppendUint64(prelude, uint64(len(jsonBytes)))
	prelude = append(prelude, jsonBytes...)
	pad := (8 - (len(prelude) % 8)) % 8
	for i := 0; i < pad; i++ {
		prelude = append(prelude, ' ')
	}

	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(prelude); err != nil {
		return err
	}
	for _, name := range sorted {
		data, shape, err := produce(name)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		ne := neByName[name]
		if want := int(PackedCliffordF16Bytes(ne)); len(data) != want {
			return fmt.Errorf("tensor %q: packed size %d != expected %d", name, len(data), want)
		}
		_ = shape // must match CliffordShape(ne); trust produce
		if _, err := f.Write(data); err != nil {
			return err
		}
	}
	return f.Close()
}

// SanitizeTensorKey maps GGUF tensor names to safetensors-safe keys.
func SanitizeTensorKey(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '.', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	s := b.String()
	if s == "" {
		s = "tensor"
	}
	if s[0] >= '0' && s[0] <= '9' {
		s = "w_" + s
	}
	return s
}
