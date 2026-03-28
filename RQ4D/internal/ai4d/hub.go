package ai4d

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Backend selects where heavy numerics run.
const (
	BackendNative = "native"
	BackendPython = "python"
)

// Hub owns the romanai.4dai model, pooling, optional Python HTTP backend, and telemetry scalars.
type Hub struct {
	mu           sync.Mutex
	M            *Model
	Pool         *FloatSlicePool
	Backend      string
	PythonBase   string
	Continuous   bool
	TrainSteps   atomic.Uint64
	InferCalls   atomic.Uint64
	InferNanos   atomic.Uint64
	SchedulerAIQ atomic.Uint64 // tasks accepted into AI tier (set by runtime)
	LastErr      atomic.Value  // string
}

// NewHub builds an empty hub (no model until LOAD).
func NewHub(backend, pythonBase string) *Hub {
	b := strings.ToLower(strings.TrimSpace(backend))
	if b == "" {
		b = BackendNative
	}
	h := &Hub{
		Pool:       NewFloatSlicePool(),
		Backend:    b,
		PythonBase: strings.TrimSuffix(strings.TrimSpace(pythonBase), "/"),
	}
	h.LastErr.Store("")
	return h
}

func (h *Hub) setErr(err error) {
	if err == nil {
		h.LastErr.Store("")
		return
	}
	h.LastErr.Store(err.Error())
}

// Load loads or replaces the in-memory model from path.
func (h *Hub) Load(path string) error {
	if h == nil {
		return fmt.Errorf("ai4d: nil hub")
	}
	m, err := LoadModel(path)
	if err != nil {
		h.setErr(err)
		return err
	}
	h.mu.Lock()
	h.M = m
	h.mu.Unlock()
	h.setErr(nil)
	return nil
}

// Save writes the current model.
func (h *Hub) Save(path string) error {
	if h == nil {
		return fmt.Errorf("ai4d: nil hub")
	}
	h.mu.Lock()
	m := h.M
	h.mu.Unlock()
	if m == nil {
		err := fmt.Errorf("ai4d: no model loaded")
		h.setErr(err)
		return err
	}
	if err := SaveModel(path, m); err != nil {
		h.setErr(err)
		return err
	}
	h.setErr(nil)
	return nil
}

// Train runs training steps (Python proxy or native toy backprop on last layer).
func (h *Hub) Train(ctx context.Context, steps int, lr float64) error {
	if h == nil || steps < 1 {
		return nil
	}
	if h.Backend == BackendPython && h.PythonBase != "" {
		return h.trainPython(ctx, steps, lr)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		err := fmt.Errorf("ai4d: no model loaded")
		h.setErr(err)
		return err
	}
	tgt := make([]float64, 0)
	if len(h.M.Layers) > 0 {
		tgt = make([]float64, h.M.Layers[len(h.M.Layers)-1].Size)
	}
	for i := 0; i < steps; i++ {
		if err := h.M.BackpropStep(tgt, lr, h.Pool); err != nil {
			h.setErr(err)
			return err
		}
		h.TrainSteps.Add(1)
	}
	h.refreshDerivedScalarsLocked()
	h.setErr(nil)
	return nil
}

type pyTrainReq struct {
	Steps int     `json:"steps"`
	LR    float64 `json:"lr"`
}

func (h *Hub) trainPython(ctx context.Context, steps int, lr float64) error {
	body, _ := json.Marshal(pyTrainReq{Steps: steps, LR: lr})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.PythonBase+"/train", bytes.NewReader(body))
	if err != nil {
		h.setErr(err)
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.setErr(err)
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		e := fmt.Errorf("python /train: %s: %s", resp.Status, string(b))
		h.setErr(e)
		return e
	}
	h.TrainSteps.Add(uint64(steps))
	h.setErr(nil)
	return nil
}

type pyInferReq struct {
	Input []float64 `json:"input"`
}

type pyInferResp struct {
	Output []float64 `json:"output"`
}

// Infer runs forward pass (native or Python).
func (h *Hub) Infer(ctx context.Context, input []float64) ([]float64, error) {
	if h == nil {
		return nil, fmt.Errorf("ai4d: nil hub")
	}
	t0 := time.Now()
	if h.Backend == BackendPython && h.PythonBase != "" {
		out, err := h.inferPython(ctx, input)
		h.InferCalls.Add(1)
		h.InferNanos.Add(uint64(time.Since(t0).Nanoseconds()))
		return out, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		err := fmt.Errorf("ai4d: no model loaded")
		h.setErr(err)
		return nil, err
	}
	out, err := h.M.ForwardPass(input, h.Pool)
	if err != nil {
		h.setErr(err)
		return nil, err
	}
	copyOut := append([]float64(nil), out...)
	if h.Pool != nil && out != nil {
		h.Pool.Put(out)
	}
	h.InferCalls.Add(1)
	h.InferNanos.Add(uint64(time.Since(t0).Nanoseconds()))
	h.setErr(nil)
	return copyOut, nil
}

func (h *Hub) inferPython(ctx context.Context, input []float64) ([]float64, error) {
	body, _ := json.Marshal(pyInferReq{Input: input})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.PythonBase+"/infer", bytes.NewReader(body))
	if err != nil {
		h.setErr(err)
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.setErr(err)
		return nil, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		e := fmt.Errorf("python /infer: %s: %s", resp.Status, string(b))
		h.setErr(e)
		return nil, e
	}
	var r pyInferResp
	if err := json.Unmarshal(b, &r); err != nil {
		h.setErr(err)
		return nil, err
	}
	h.setErr(nil)
	return r.Output, nil
}

// RotateFirstLayer left-multiplies every 4×4 block by a plane rotation (Roma4D ROTATE).
func (h *Hub) Rotate(plane string, angle float64) error {
	if h == nil {
		return fmt.Errorf("ai4d: nil hub")
	}
	R := Mat4RotatePlane(plane, angle)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil || len(h.M.Layers) == 0 {
		err := fmt.Errorf("ai4d: no model loaded")
		h.setErr(err)
		return err
	}
	L := &h.M.Layers[0]
	for b := range L.Blocks {
		L.Blocks[b] = Mat4Mul(R, L.Blocks[b])
	}
	h.refreshDerivedScalarsLocked()
	h.setErr(nil)
	return nil
}

// EvolveQuantum relaxes header scalar hints (deterministic Ricci-flow style toy on stored hints only).
func (h *Hub) EvolveQuantum(steps int) error {
	if h == nil || steps < 1 {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		err := fmt.Errorf("ai4d: no model loaded")
		h.setErr(err)
		return err
	}
	dt := 0.01
	for i := 0; i < steps; i++ {
		h.M.Header.ManifoldStrain *= 1 - dt*0.05
		h.M.Header.RicciRelaxation += dt * 0.02
		if h.M.Header.RicciRelaxation > 1 {
			h.M.Header.RicciRelaxation = 1
		}
	}
	h.setErr(nil)
	return nil
}

func (h *Hub) refreshDerivedScalarsLocked() {
	if h.M == nil {
		return
	}
	var sum float64
	n := 0
	for _, L := range h.M.Layers {
		for _, b := range L.Blocks {
			for _, v := range b {
				sum += v * v
				n++
			}
		}
	}
	if n > 0 {
		h.M.Header.ManifoldStrain = sum / float64(n)
	}
}

// SnapshotModelHeader returns a copy of header + layer sizes (for metrics).
func (h *Hub) SnapshotModelHeader() (FileHeader, []int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		return FileHeader{}, nil
	}
	sz := make([]int, len(h.M.Layers))
	for i := range h.M.Layers {
		sz[i] = h.M.Layers[i].Size
	}
	return h.M.Header, sz
}

// CognitiveGravityWell returns a deterministic scalar from weight topology density.
func (h *Hub) CognitiveGravityWell() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		return 0
	}
	var sum float64
	for _, L := range h.M.Layers {
		sum += float64(len(L.Blocks))
	}
	return sum * (1 + h.M.Header.ManifoldStrain)
}

// WAxisContraction returns the model's W contraction factor for 3D exports.
func (h *Hub) WAxisContraction() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		return 1
	}
	return h.M.Header.WContraction
}

// SetWContraction updates perspective contraction (thread-safe).
func (h *Hub) SetWContraction(w float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M == nil {
		return
	}
	h.M.Header.WContraction = w
}

// Export3DProjects applies W contraction from current model header.
func (h *Hub) Export3DProjects(vec []float64) []float64 {
	h.mu.Lock()
	w := 1.0
	if h.M != nil {
		w = h.M.Header.WContraction
	}
	h.mu.Unlock()
	return Export3D(vec, w)
}

// InitCliffordLayer replaces the loaded model with a fresh Clifford layer of the given width.
func (h *Hub) InitCliffordLayer(size int) error {
	if h == nil {
		return fmt.Errorf("ai4d: nil hub")
	}
	m, err := NewModelClifford(size)
	if err != nil {
		h.setErr(err)
		return err
	}
	h.mu.Lock()
	h.M = m
	h.refreshDerivedScalarsLocked()
	h.mu.Unlock()
	h.setErr(nil)
	return nil
}

// EnsureModel creates an empty Clifford model if none loaded (for INFER smoke tests).
func (h *Hub) EnsureModel(defaultSize int) error {
	if h == nil {
		return fmt.Errorf("ai4d: nil hub")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.M != nil {
		return nil
	}
	if defaultSize < 4 {
		defaultSize = 4
	}
	m, err := NewModelClifford(defaultSize)
	if err != nil {
		return err
	}
	h.M = m
	return nil
}
