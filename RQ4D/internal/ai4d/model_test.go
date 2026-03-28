package ai4d

import (
	"bytes"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"
)

func TestModelRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "t.4dai")
	m, err := NewModelClifford(8)
	if err != nil {
		t.Fatal(err)
	}
	m.Header.ManifoldStrain = 0.25
	m.Header.WContraction = 0.9
	if err := SaveModel(p, m); err != nil {
		t.Fatal(err)
	}
	m2, err := LoadModel(p)
	if err != nil {
		t.Fatal(err)
	}
	if m2.Header.WContraction != 0.9 || len(m2.Layers) != 1 || m2.Layers[0].Size != 8 {
		t.Fatalf("%+v", m2)
	}
}

func TestForwardPassIdentity(t *testing.T) {
	m, _ := NewModelClifford(4)
	pool := NewFloatSlicePool()
	out, err := m.ForwardPass([]float64{1, 2, 3, 4}, pool)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Put(out)
	want := []float64{1, 2, 3, 4}
	for i := range want {
		if math.Abs(out[i]-want[i]) > 1e-9 {
			t.Fatalf("got %v want %v", out, want)
		}
	}
}

func TestGGUFManifestJSON(t *testing.T) {
	m, _ := NewModelClifford(4)
	var buf bytes.Buffer
	if err := m.WriteGGUFManifestJSON(&buf); err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(buf.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
}

func TestExport3D(t *testing.T) {
	v := []float64{1, 0, 0, 2}
	out := Export3D(v, 0.5)
	if len(out) != 3 {
		t.Fatalf("%v", out)
	}
}
