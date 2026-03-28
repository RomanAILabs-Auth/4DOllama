package rq4dcore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/ai4d"
)

// BridgeServer exposes a loopback-safe HTTP API (no Ollama binary changes, no hooks).
type BridgeServer struct {
	rt         *Runtime
	prof       *Profiler
	ollamaBase string
	srv        *http.Server
}

// NewBridgeServer validates addr is loopback unless allowNonLoopback is true.
func NewBridgeServer(rt *Runtime, prof *Profiler, addr string, allowNonLoopback bool, ollamaBase string) (*BridgeServer, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, fmt.Errorf("bridge: invalid listen addr %q: %w", addr, err)
	}
	if !allowNonLoopback {
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return nil, fmt.Errorf("bridge: listen %q is not loopback (use 127.0.0.1 or set --bridge-allow-non-loopback)", addr)
		}
	}
	b := &BridgeServer{rt: rt, prof: prof, ollamaBase: strings.TrimSuffix(ollamaBase, "/")}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", b.handleHealth)
	mux.HandleFunc("/v1/rq4d/diagnostics", b.handleDiagnostics)
	mux.HandleFunc("/v1/rq4d/step", b.handleStep)
	mux.HandleFunc("/v1/rq4d/measure", b.handleMeasure)
	mux.HandleFunc("/v1/rq4d/transform", b.handleTransform)
	mux.HandleFunc("/v1/metrics", b.handleMetrics)
	if b.ollamaBase != "" {
		mux.HandleFunc("/v1/ollama/forward", b.handleOllamaForward)
	}
	if b.rt != nil && b.rt.AI != nil {
		mux.HandleFunc("/v1/ai4d/status", b.handleAI4DStatus)
		mux.HandleFunc("/v1/ai4d/metrics", b.handleAI4DMetrics)
		mux.HandleFunc("/v1/ai4d/exec", b.handleAI4DExec)
		mux.HandleFunc("/v1/ai4d/train", b.handleAI4DTrain)
		mux.HandleFunc("/v1/ai4d/infer", b.handleAI4DInfer)
		mux.HandleFunc("/train", b.handleAI4DTrain)
		mux.HandleFunc("/infer", b.handleAI4DInfer)
	}
	b.srv = &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	return b, nil
}

func (b *BridgeServer) ListenAndServe() error {
	return b.srv.ListenAndServe()
}

func (b *BridgeServer) Shutdown(ctx context.Context) error {
	if b.srv == nil {
		return nil
	}
	return b.srv.Shutdown(ctx)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (b *BridgeServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, `{"status":"ok","service":"rq4d-ollama-bridge"}`)
}

func (b *BridgeServer) handleDiagnostics(w http.ResponseWriter, _ *http.Request) {
	norm2, expectH, sha := b.rt.Eng.SnapshotDiagnostics()
	writeJSON(w, http.StatusOK, map[string]any{
		"norm2": norm2, "expectation_H_mf": expectH, "state_sha256": sha,
	})
}

type stepReq struct {
	Steps int `json:"steps"`
}

func (b *BridgeServer) handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req stepReq
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.Steps < 1 {
		req.Steps = 1
	}
	resp := make(chan TaskResult, 1)
	t := Task{Kind: TaskLatticeStep, Steps: req.Steps, Resp: resp}
	if !b.rt.SubmitSim(t) {
		http.Error(w, "sim queue saturated", http.StatusServiceUnavailable)
		return
	}
	select {
	case res := <-resp:
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "norm2": res.Norm2, "expectation_H_mf": res.ExpectH, "state_sha256": res.StateSHA})
	case <-time.After(30 * time.Second):
		http.Error(w, "timeout", http.StatusGatewayTimeout)
	}
}

type measureReq struct {
	Site int   `json:"site"`
	Seed int64 `json:"seed"`
}

func (b *BridgeServer) handleMeasure(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req measureReq
	_ = json.NewDecoder(r.Body).Decode(&req)
	resp := make(chan TaskResult, 1)
	t := Task{Kind: TaskMeasurementSample, Site: req.Site, Seed: req.Seed, Resp: resp}
	if !b.rt.SubmitSim(t) {
		http.Error(w, "sim queue saturated", http.StatusServiceUnavailable)
		return
	}
	select {
	case res := <-resp:
		writeJSON(w, http.StatusOK, map[string]any{
			"ok": true, "outcome": res.MeasureK, "norm2": res.Norm2, "expectation_H_mf": res.ExpectH, "state_sha256": res.StateSHA,
		})
	case <-time.After(30 * time.Second):
		http.Error(w, "timeout", http.StatusGatewayTimeout)
	}
}

type transformReq struct {
	Prompt      string `json:"prompt"`
	WarmupSteps int    `json:"warmup_steps"`
}

func (b *BridgeServer) handleTransform(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req transformReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	resp := make(chan TaskResult, 1)
	t := Task{Kind: TaskExternalRequest, Prompt: req.Prompt, WarmupSteps: req.WarmupSteps, Resp: resp}
	if !b.rt.SubmitExternal(t) {
		http.Error(w, "external queue saturated", http.StatusServiceUnavailable)
		return
	}
	select {
	case res := <-resp:
		writeJSON(w, http.StatusOK, map[string]any{
			"ok": true, "transform": res.Transform, "norm2": res.Norm2, "expectation_H_mf": res.ExpectH, "state_sha256": res.StateSHA,
		})
	case <-time.After(30 * time.Second):
		http.Error(w, "timeout", http.StatusGatewayTimeout)
	}
}

func (b *BridgeServer) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	if b.prof == nil {
		writeJSON(w, http.StatusOK, map[string]any{})
		return
	}
	m := b.prof.Snapshot()
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	writeJSON(w, http.StatusOK, out)
}

type ollamaForwardReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

func (b *BridgeServer) handleOllamaForward(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req ollamaForwardReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model == "" || req.Prompt == "" {
		http.Error(w, "need json {model, prompt}", http.StatusBadRequest)
		return
	}
	body, _ := json.Marshal(map[string]any{
		"model":  req.Model,
		"prompt": req.Prompt,
		"stream": false,
	})
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()
	url := b.ollamaBase + "/api/generate"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		http.Error(w, string(data), resp.StatusCode)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (b *BridgeServer) handleAI4DStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	if b.rt == nil || b.rt.AI == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false, "error": "ai hub off"})
		return
	}
	h, layers := b.rt.AI.SnapshotModelHeader()
	n2, eh, sha := b.rt.Eng.SnapshotDiagnostics()
	lastErr := ""
	if v := b.rt.AI.LastErr.Load(); v != nil {
		if s, ok := v.(string); ok {
			lastErr = s
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                         true,
		"service":                    "ai4d",
		"ai_backend":                 b.rt.AI.Backend,
		"continuous_execution":       b.rt.AI.Continuous,
		"model_format":               h.Format,
		"model_version":              h.Version,
		"layer_sizes":                layers,
		"w_axis_contraction":         b.rt.AI.WAxisContraction(),
		"manifold_strain":            h.ManifoldStrain,
		"ricci_flow_relaxation":      h.RicciRelaxation,
		"cognitive_gravity_well":     b.rt.AI.CognitiveGravityWell(),
		"lattice_norm2":              n2,
		"lattice_expectation_H_mf":   eh,
		"lattice_state_sha256":       sha,
		"scheduler_ai_tasks_queued":  b.rt.AI.SchedulerAIQ.Load(),
		"last_error":                 lastErr,
	})
}

func (b *BridgeServer) handleAI4DMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET only", http.StatusMethodNotAllowed)
		return
	}
	if b.rt == nil || b.rt.AI == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false})
		return
	}
	h, _ := b.rt.AI.SnapshotModelHeader()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	infCalls := b.rt.AI.InferCalls.Load()
	infNs := b.rt.AI.InferNanos.Load()
	var avgInfer float64
	if infCalls > 0 {
		avgInfer = float64(infNs) / float64(infCalls) / 1e6
	}
	trainSteps := b.rt.AI.TrainSteps.Load()
	out := map[string]any{
		"ok":                             true,
		"inference_total_calls":          infCalls,
		"inference_avg_latency_ms":       avgInfer,
		"training_steps_completed":       trainSteps,
		"manifold_strain":                h.ManifoldStrain,
		"ricci_flow_relaxation":          h.RicciRelaxation,
		"w_axis_perspective_contraction": b.rt.AI.WAxisContraction(),
		"cognitive_gravity_well_depth":   b.rt.AI.CognitiveGravityWell(),
		"mem_heap_alloc_bytes":           m.HeapAlloc,
		"mem_sys_bytes":                  m.Sys,
	}
	if b.prof != nil {
		s := b.prof.Snapshot()
		for k, v := range s {
			out["scheduler_"+k] = v
		}
	}
	writeJSON(w, http.StatusOK, out)
}

type ai4dExecReq struct {
	Script string `json:"script"`
}

func (b *BridgeServer) handleAI4DExec(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if b.rt == nil || b.rt.AI == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false, "error": "ai hub off"})
		return
	}
	var req ai4dExecReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	cmds, err := ai4d.ParseScript(req.Script)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"ok": false, "error": err.Error()})
		return
	}
	var results []map[string]any
	for _, c := range cmds {
		t, ok := TaskFromAICommand(c)
		if !ok {
			continue
		}
		resp := make(chan TaskResult, 1)
		t.Resp = resp
		if !b.rt.SubmitAI(t) {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false, "error": "ai queue saturated", "partial": results})
			return
		}
		select {
		case res := <-resp:
			entry := map[string]any{"ok": res.Err == nil}
			if res.Err != nil {
				entry["error"] = res.Err.Error()
			}
			if res.AIOut != nil {
				entry["infer_out"] = res.AIOut
			}
			results = append(results, entry)
		case <-time.After(5 * time.Minute):
			writeJSON(w, http.StatusGatewayTimeout, map[string]any{"ok": false, "error": "timeout", "partial": results})
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "results": results})
}

type ai4dTrainHTTPReq struct {
	Steps int     `json:"steps"`
	LR    float64 `json:"lr"`
}

func (b *BridgeServer) handleAI4DTrain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if b.rt == nil || b.rt.AI == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false})
		return
	}
	var req ai4dTrainHTTPReq
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.Steps < 1 {
		req.Steps = 1
	}
	if req.LR == 0 {
		req.LR = 0.001
	}
	resp := make(chan TaskResult, 1)
	t := Task{Kind: TaskAITrainStep, Steps: req.Steps, AILR: req.LR, Resp: resp}
	if !b.rt.SubmitAI(t) {
		http.Error(w, "ai queue saturated", http.StatusServiceUnavailable)
		return
	}
	select {
	case res := <-resp:
		if res.Err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": res.Err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	case <-time.After(30 * time.Minute):
		http.Error(w, "timeout", http.StatusGatewayTimeout)
	}
}

type ai4dInferHTTPReq struct {
	Input []float64 `json:"input"`
}

func (b *BridgeServer) handleAI4DInfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if b.rt == nil || b.rt.AI == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false})
		return
	}
	var req ai4dInferHTTPReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	body, _ := json.Marshal(req.Input)
	resp := make(chan TaskResult, 1)
	t := Task{Kind: TaskAIInference, AIBody: string(body), Resp: resp}
	if !b.rt.SubmitAI(t) {
		http.Error(w, "ai queue saturated", http.StatusServiceUnavailable)
		return
	}
	select {
	case res := <-resp:
		if res.Err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"ok": false, "error": res.Err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "output": res.AIOut})
	case <-time.After(2 * time.Minute):
		http.Error(w, "timeout", http.StatusGatewayTimeout)
	}
}
