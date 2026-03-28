package ai4d

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	FormatID   = "romanai.4dai"
	FormatVer1 = 1
)

// FileHeader is the JSON preamble for .4dai files (human-auditable + deterministic).
type FileHeader struct {
	Format          string  `json:"format"`
	Version         int     `json:"version"`
	WContraction    float64 `json:"w_axis_perspective_contraction"`
	ManifoldStrain  float64 `json:"manifold_strain_hint"`
	RicciRelaxation float64 `json:"ricci_flow_relaxation_hint"`
}

// LayerSpec describes one executable layer block.
type LayerSpec struct {
	Kind   string    `json:"kind"`
	Size   int       `json:"size"`
	Blocks []float64 `json:"blocks,omitempty"` // size/4 matrices, row-major 16 floats each
}

// ModelFile is the on-disk JSON envelope.
type ModelFile struct {
	Header FileHeader  `json:"header"`
	Layers []LayerSpec `json:"layers"`
}

// Model is an in-memory romanai.4dai view. Weights are 4×4 blocks (16 floats) acting on 4-chunks of activations.
type Model struct {
	Header FileHeader
	Layers []CliffordLayer
}

// CliffordLayer holds per-block 4×4 matrices (Cl(4,0) real matrix representation on 4D fibers).
type CliffordLayer struct {
	Size   int
	Blocks []Mat4 // len = Size/4
}

// IdentityMat4 returns I₄.
func IdentityMat4() Mat4 {
	return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
}

// LoadModel reads a .4dai JSON file.
func LoadModel(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseModelBytes(b)
}

// ParseModelBytes decodes JSON model bytes.
func ParseModelBytes(b []byte) (*Model, error) {
	var mf ModelFile
	if err := json.Unmarshal(b, &mf); err != nil {
		return nil, err
	}
	if mf.Header.Format != FormatID {
		return nil, fmt.Errorf("ai4d: expected format %q, got %q", FormatID, mf.Header.Format)
	}
	if mf.Header.Version != FormatVer1 {
		return nil, fmt.Errorf("ai4d: unsupported version %d", mf.Header.Version)
	}
	m := &Model{Header: mf.Header}
	for _, ls := range mf.Layers {
		if ls.Kind != "" && ls.Kind != "clifford" {
			return nil, fmt.Errorf("ai4d: unknown layer kind %q", ls.Kind)
		}
		if ls.Size < 4 || ls.Size%4 != 0 {
			return nil, fmt.Errorf("ai4d: layer size must be multiple of 4, got %d", ls.Size)
		}
		nBlk := ls.Size / 4
		if len(ls.Blocks) != nBlk*16 {
			return nil, fmt.Errorf("ai4d: blocks length want %d got %d", nBlk*16, len(ls.Blocks))
		}
		cl := CliffordLayer{Size: ls.Size, Blocks: make([]Mat4, nBlk)}
		for i := 0; i < nBlk; i++ {
			copy(cl.Blocks[i][:], ls.Blocks[i*16:(i+1)*16])
		}
		m.Layers = append(m.Layers, cl)
	}
	return m, nil
}

// SaveModel writes JSON .4dai (deterministic: standard library json with stable layer order).
func SaveModel(path string, m *Model) error {
	if m == nil {
		return errors.New("ai4d: nil model")
	}
	mf := ModelFile{Header: m.Header, Layers: nil}
	for _, L := range m.Layers {
		ls := LayerSpec{Kind: "clifford", Size: L.Size, Blocks: make([]float64, 0, len(L.Blocks)*16)}
		for _, blk := range L.Blocks {
			ls.Blocks = append(ls.Blocks, blk[:]...)
		}
		mf.Layers = append(mf.Layers, ls)
	}
	b, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// NewModelClifford builds a single Clifford layer with identity blocks (size multiple of 4).
func NewModelClifford(size int) (*Model, error) {
	if size < 4 || size%4 != 0 {
		return nil, fmt.Errorf("ai4d: size must be >=4 and multiple of 4")
	}
	n := size / 4
	blocks := make([]Mat4, n)
	for i := range blocks {
		blocks[i] = IdentityMat4()
	}
	return &Model{
		Header: FileHeader{
			Format: FormatID, Version: FormatVer1,
			WContraction: 1, ManifoldStrain: 0, RicciRelaxation: 0,
		},
		Layers: []CliffordLayer{{Size: size, Blocks: blocks}},
	}, nil
}

// ForwardPass runs activations through all layers (chunked 4×4, no cross-chunk mixing — lightweight baseline).
func (m *Model) ForwardPass(in []float64, pool *FloatSlicePool) ([]float64, error) {
	if m == nil || len(m.Layers) == 0 {
		out := append([]float64(nil), in...)
		return out, nil
	}
	x := pool.Get(len(in))
	copy(x, in)
	for li := range m.Layers {
		L := &m.Layers[li]
		if len(x) != L.Size && li == 0 {
			// First layer defines working width; pad or trim for determinism.
			if len(x) < L.Size {
				nx := pool.Get(L.Size)
				copy(nx, x)
				pool.Put(x)
				x = nx
			} else {
				x = x[:L.Size]
			}
		} else if len(x) != L.Size {
			return nil, fmt.Errorf("ai4d: dim mismatch layer %d want %d got %d", li, L.Size, len(x))
		}
		next := pool.Get(L.Size)
		for b := 0; b < len(L.Blocks); b++ {
			var v Vec4
			for j := 0; j < 4; j++ {
				v[j] = x[b*4+j]
			}
			o := Mul4(L.Blocks[b], v)
			for j := 0; j < 4; j++ {
				next[b*4+j] = o[j]
			}
		}
		pool.Put(x)
		x = next
	}
	return x, nil
}

// BackpropStep performs one SGD-like step on the last layer blocks (minimal, deterministic toy rule).
func (m *Model) BackpropStep(target []float64, lr float64, pool *FloatSlicePool) error {
	if m == nil || len(m.Layers) == 0 {
		return nil
	}
	L := &m.Layers[len(m.Layers)-1]
	if len(target) != L.Size {
		return fmt.Errorf("ai4d: backprop target len %d != layer %d", len(target), L.Size)
	}
	// Synthetic "forward" state: reuse ForwardPass from zeros would be odd; instead nudge blocks toward projecting toward target on identity path.
	for b := 0; b < len(L.Blocks); b++ {
		var v Vec4
		for j := 0; j < 4; j++ {
			v[j] = target[b*4+j]
		}
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				idx := i*4 + j
				// Very small corrective step: pull diagonal toward target correlation (deterministic).
				delta := lr * v[i] * v[j]
				if i == j {
					delta *= 0.25
				}
				L.Blocks[b][idx] -= delta
			}
		}
	}
	_ = pool
	return nil
}

// Export3D applies W-axis perspective contraction and drops the W component (index 3 of each 4-block).
func Export3D(vec []float64, wContraction float64) []float64 {
	if len(vec)%4 != 0 {
		out := make([]float64, 0, len(vec)*3/4)
		for i := 0; i+3 < len(vec); i += 4 {
			w := vec[i+3] * wContraction
			out = append(out, vec[i]+0.25*w, vec[i+1]+0.25*w, vec[i+2]+0.25*w)
		}
		return out
	}
	out := make([]float64, 0, len(vec)/4*3)
	for i := 0; i < len(vec); i += 4 {
		w := vec[i+3] * wContraction
		out = append(out, vec[i]+0.25*w, vec[i+1]+0.25*w, vec[i+2]+0.25*w)
	}
	return out
}

// GGUFManifest describes a future binary GGUF mapping (JSON sidecar for Ollama ingestion planning).
type GGUFManifest struct {
	Note     string            `json:"note"`
	Tensors  []GGUFTensorDesc  `json:"tensors"`
	Metadata map[string]string `json:"metadata"`
}

type GGUFTensorDesc struct {
	Name   string `json:"name"`
	Shape  []int  `json:"shape"`
	DType  string `json:"dtype"`
	Offset int64  `json:"offset_in_blob"`
	Bytes  int64  `json:"byte_length"`
}

// WriteGGUFManifestJSON writes a sidecar description; does not emit full binary GGUF.
func (m *Model) WriteGGUFManifestJSON(w io.Writer) error {
	if m == nil {
		return errors.New("ai4d: nil model")
	}
	var tensors []GGUFTensorDesc
	off := int64(0)
	for i, L := range m.Layers {
		for j := range L.Blocks {
			name := fmt.Sprintf("layer_%d.block_%d.weight", i, j)
			tensors = append(tensors, GGUFTensorDesc{
				Name: name, Shape: []int{4, 4}, DType: "F32", Offset: off, Bytes: 4 * 16,
			})
			off += 4 * 16
		}
	}
	g := GGUFManifest{
		Note: "Experimental layout: 4×4 Cl(4,0) blocks as f32 row-major; wrap with external tooling for full GGUF.",
		Tensors: tensors,
		Metadata: map[string]string{
			"general.architecture": "romanai.4dai",
			"general.format":       FormatID,
		},
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(g)
}

// GGUFBlobAppend writes raw f32 little-endian weights for manifest offsets (optional packer).
func (m *Model) GGUFBlobAppend(buf *bytes.Buffer) error {
	if m == nil {
		return errors.New("ai4d: nil model")
	}
	for _, L := range m.Layers {
		for _, blk := range L.Blocks {
			for _, f := range blk {
				var b [4]byte
				binary.LittleEndian.PutUint32(b[:], math.Float32bits(float32(f)))
				if _, err := buf.Write(b[:]); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
