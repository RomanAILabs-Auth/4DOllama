//go:build !cgo

package engine

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/4dollama/4dollama/internal/models"
	"github.com/4dollama/4dollama/internal/version"
)

type stubEngine struct{}

// New constructs the engine. Without cgo, inspection is unavailable.
func New() Engine {
	return stubEngine{}
}

func (stubEngine) Info() Info {
	return Info{Version: version.Version, Backend: BackendStub}
}

func (stubEngine) Capabilities() ([]byte, error) {
	gpu := stubActiveGPUBackend()
	if gpu == "" {
		gpu = "cpu"
	}
	return []byte(fmt.Sprintf(
		`{"name":"four_d_engine","version":"stub","abi_version":1,"computational_model":{"tensor_rank":4,"note":"Rebuild with CGO_ENABLED=1 after cargo build --release for native quaternion/4D core."},"ffi":{"gguf_inspect_json":false,"capabilities_json":false,"sample_next_token_4d":true,"gemm4d":true,"gpu_backend":"%s","autoregressive_inference":true},"prior_art":"See docs/PRIOR_ART.md"}`,
		gpu,
	)), nil
}

func (stubEngine) GGUFParamCount(path string) (int64, error) {
	return models.CountGGUFParams(path)
}

func (stubEngine) GGUFSampleLift(string, int) ([]float32, int64, error) {
	return nil, 0, fmt.Errorf("GGUFSampleLift requires CGO native engine (cargo build + CGO_ENABLED=1)")
}

func (stubEngine) Compute4DDemo(in []float32) ([]float32, error) {
	return compute4DDemoGo(in), nil
}

func (stubEngine) Rope4DSequence(in []float32) ([]float32, error) {
	return rope4DSequenceGo(in), nil
}

func (stubEngine) SpacetimeAttention4D(q, k, v []float32, seqLen int) ([]float32, error) {
	if stubActiveGPUBackend() != "" {
		return spacetimeAttentionParallelGo(q, k, v, seqLen), nil
	}
	return spacetimeAttentionGo(q, k, v, seqLen), nil
}

func (stubEngine) SampleNextToken4D(logits []float32, temperature float32, topK int) (uint32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("SampleNextToken4D: empty logits")
	}
	if topK < 0 {
		topK = 0
	}
	return sampleNextToken4DGo(logits, temperature, topK), nil
}

func (stubEngine) ProjectStubLogits(last []float32, lifted []float32, vocabSize int, step int, logFirst bool) ([]float32, error) {
	out := projectStubLogitsGo(last, lifted, vocabSize, step, logFirst)
	if out == nil {
		return nil, fmt.Errorf("ProjectStubLogits: invalid args")
	}
	return out, nil
}

func (stubEngine) SampleNextTokenFlat(logits []float32, temperature float32, topK int) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("SampleNextTokenFlat: empty logits")
	}
	if topK < 0 {
		topK = 0
	}
	return sampleNextTokenFlatGo(logits, temperature, topK), nil
}

func (stubEngine) Gemm4D(a, b []float32, m, k, n int) ([]float32, error) {
	if m <= 0 || k <= 0 || n <= 0 {
		return nil, fmt.Errorf("Gemm4D: invalid dims")
	}
	needA, needB := m*k, k*n
	if len(a) < needA || len(b) < needB {
		return nil, fmt.Errorf("Gemm4D: buffer too small")
	}
	if stubActiveGPUBackend() != "" {
		return gemm4dGoParallel(a[:needA], b[:needB], m, k, n), nil
	}
	return gemm4dGo(a[:needA], b[:needB], m, k, n), nil
}

func (stubEngine) GPUBackend() string {
	if s := stubActiveGPUBackend(); s != "" {
		return s
	}
	return "cpu"
}

func stubActiveGPUBackend() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FOURD_GPU"))) {
	case "0", "off", "cpu", "none":
		return ""
	}
	if runtime.GOOS == "darwin" {
		if _, err := os.Stat("/System/Library/Frameworks/Metal.framework"); err == nil {
			return "metal"
		}
	}
	for _, p := range []string{
		"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
		"/usr/lib/wsl/lib/libcuda.so",
	} {
		if _, err := os.Stat(p); err == nil {
			return "cuda"
		}
	}
	if runtime.GOOS == "windows" && strings.TrimSpace(os.Getenv("CUDA_PATH")) != "" {
		return "cuda"
	}
	return ""
}

func compute4DDemoGo(input []float32) []float32 {
	if len(input) == 0 {
		return nil
	}
	var seed float32
	for _, b := range input {
		if b < 0 {
			seed -= b
		} else {
			seed += b
		}
	}
	angle := float32(math.Sin(float64(seed*0.1)) * math.Pi)
	half := angle * 0.5
	ch := float32(math.Cos(float64(half)))
	sh := float32(math.Sin(float64(half)))
	ax := float32(1)
	ay := float32(0)
	az := float32(0)
	if len(input) > 0 {
		ax = input[0]
	}
	if len(input) > 1 {
		ay = input[1]
	}
	if len(input) > 2 {
		az = input[2]
	}
	n := float32(math.Sqrt(float64(ax*ax + ay*ay + az*az)))
	if n <= 1e-6 {
		ax, ay, az = 1, 0, 0
	} else {
		ax /= n
		ay /= n
		az /= n
	}
	qw, qi, qj, qk := ch, ax*sh, ay*sh, az*sh
	rotate := func(v [3]float32) [3]float32 {
		pw, pi, pj, pk := float32(0), v[0], v[1], v[2]
		rw := qw*pw - qi*pi - qj*pj - qk*pk
		ri := qw*pi + qi*pw + qj*pk - qk*pj
		rj := qw*pj - qi*pk + qj*pw + qk*pi
		rk := qw*pk + qi*pj - qj*pi + qk*pw
		n2 := qw*qw + qi*qi + qj*qj + qk*qk
		if n2 <= 1e-12 {
			return v
		}
		iqw, iqi, iqj, iqk := qw/n2, -qi/n2, -qj/n2, -qk/n2
		fw := rw*iqw - ri*iqi - rj*iqj - rk*iqk
		fi := rw*iqi + ri*iqw + rj*iqk - rk*iqj
		fj := rw*iqj - ri*iqk + rj*iqw + rk*iqi
		fk := rw*iqk + ri*iqj - rj*iqi + rk*iqw
		_ = fw
		return [3]float32{fi, fj, fk}
	}
	v := append([]float32(nil), input...)
	for len(v)%3 != 0 {
		v = append(v, 0)
	}
	out := make([]float32, 0, len(v))
	for i := 0; i < len(v); i += 3 {
		r := rotate([3]float32{v[i], v[i+1], v[i+2]})
		out = append(out, r[0], r[1], r[2])
	}
	return out
}

func quatMul(a, b [4]float32) [4]float32 {
	return [4]float32{
		a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
		a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
		a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
		a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
	}
}

func quatConj(q [4]float32) [4]float32 {
	return [4]float32{q[0], -q[1], -q[2], -q[3]}
}

func quatNormSq(q [4]float32) float32 {
	return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
}

func quatNormalize(q [4]float32) [4]float32 {
	n := float32(math.Sqrt(float64(quatNormSq(q))))
	if n <= 1e-6 {
		return [4]float32{1, 0, 0, 0}
	}
	return [4]float32{q[0] / n, q[1] / n, q[2] / n, q[3] / n}
}

func applyQuatRopeChunk(q [4]float32, position int) [4]float32 {
	pos := float32(position)
	theta := pos * 0.07
	axis := [3]float32{
		float32(math.Sin(float64(pos*0.31 + 0.1))),
		float32(math.Cos(float64(pos*0.27 + 0.2))),
		float32(math.Sin(float64(pos*0.19 + 0.3))),
	}
	an := float32(math.Sqrt(float64(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])))
	if an < 1e-6 {
		an = 1e-6
	}
	h := theta * 0.5
	ch := float32(math.Cos(float64(h)))
	sh := float32(math.Sin(float64(h)))
	r := quatNormalize([4]float32{
		ch,
		axis[0] / an * sh,
		axis[1] / an * sh,
		axis[2] / an * sh,
	})
	qv := quatNormalize(q)
	rot := quatMul(quatMul(r, qv), quatConj(r))
	return rot
}

func rope4DSequenceGo(v []float32) []float32 {
	if len(v) == 0 {
		return nil
	}
	x := append([]float32(nil), v...)
	for len(x)%4 != 0 {
		x = append(x, 0)
	}
	out := make([]float32, 0, len(x))
	for i := 0; i < len(x); i += 4 {
		rot := applyQuatRopeChunk([4]float32{x[i], x[i+1], x[i+2], x[i+3]}, i/4)
		out = append(out, rot[0], rot[1], rot[2], rot[3])
	}
	return out
}

func quatAtNorm(buf []float32, token int) [4]float32 {
	o := token * 4
	return quatNormalize([4]float32{buf[o], buf[o+1], buf[o+2], buf[o+3]})
}

func quatDot4(a, b [4]float32) float32 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

func softmax32(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	m := logits[0]
	for _, x := range logits[1:] {
		if x > m {
			m = x
		}
	}
	exps := make([]float32, len(logits))
	var sum float32
	for i, x := range logits {
		exps[i] = float32(math.Exp(float64(x - m)))
		sum += exps[i]
	}
	if sum <= 1e-12 {
		u := 1.0 / float32(len(logits))
		for i := range exps {
			exps[i] = u
		}
		return exps
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

func spacetimeRowAtGo(q, k, v []float32, seqLen, i int) [4]float32 {
	qi := quatAtNorm(q, i)
	logits := make([]float32, 0, i+1)
	for j := 0; j <= i; j++ {
		kj := quatAtNorm(k, j)
		dot := quatDot4(qi, kj)
		dt := float32(i - j)
		score := dot - 0.18*dt*dt
		logits = append(logits, score*1.25)
	}
	w := softmax32(logits)
	var acc [4]float32
	for j := 0; j <= i; j++ {
		vj := quatAtNorm(v, j)
		ww := w[j]
		acc[0] += ww * vj[0]
		acc[1] += ww * vj[1]
		acc[2] += ww * vj[2]
		acc[3] += ww * vj[3]
	}
	mixed := applyQuatRopeChunk(acc, i)
	return quatNormalize([4]float32{mixed[0], mixed[1], mixed[2], mixed[3]})
}

func spacetimeAttentionGo(q, k, v []float32, seqLen int) []float32 {
	need := seqLen * 4
	if seqLen <= 0 || len(q) < need || len(k) < need || len(v) < need {
		return nil
	}
	out := make([]float32, need)
	for i := 0; i < seqLen; i++ {
		nq := spacetimeRowAtGo(q, k, v, seqLen, i)
		o := i * 4
		out[o], out[o+1], out[o+2], out[o+3] = nq[0], nq[1], nq[2], nq[3]
	}
	return out
}

func spacetimeAttentionParallelGo(q, k, v []float32, seqLen int) []float32 {
	need := seqLen * 4
	if seqLen <= 0 || len(q) < need || len(k) < need || len(v) < need {
		return nil
	}
	nt := min(8, max(1, runtime.NumCPU()))
	if seqLen < 12 || nt <= 1 {
		return spacetimeAttentionGo(q, k, v, seqLen)
	}
	out := make([]float32, need)
	var wg sync.WaitGroup
	chunk := (seqLen + nt - 1) / nt
	for t := 0; t < nt; t++ {
		i0 := t * chunk
		i1 := min(i0+chunk, seqLen)
		if i0 >= i1 {
			continue
		}
		wg.Add(1)
		go func(i0, i1 int) {
			defer wg.Done()
			for i := i0; i < i1; i++ {
				nq := spacetimeRowAtGo(q, k, v, seqLen, i)
				o := i * 4
				out[o], out[o+1], out[o+2], out[o+3] = nq[0], nq[1], nq[2], nq[3]
			}
		}(i0, i1)
	}
	wg.Wait()
	return out
}

func spacetimeAttentionDispatchStub(q, k, v []float32, seqLen int) []float32 {
	if stubActiveGPUBackend() != "" {
		return spacetimeAttentionParallelGo(q, k, v, seqLen)
	}
	return spacetimeAttentionGo(q, k, v, seqLen)
}

func readoutQuat4() [4]float32 {
	return quatNormalize([4]float32{0.72, 0.45, -0.38, 0.29})
}

func detSampleIndexGo(probs []float32, logits []float32, temperature float32, topK int) int {
	var h uint64 = 1469598103934665603
	for _, x := range logits {
		h ^= uint64(math.Float32bits(x))
		h *= 1099511628211
	}
	h ^= bits.RotateLeft64(uint64(topK), 17)
	h ^= bits.RotateLeft64(uint64(math.Float32bits(temperature)), 31)
	r := float32(h%1000000) / 1e6
	var c float32
	for i, p := range probs {
		c += p
		if r < c {
			return i
		}
	}
	if len(probs) == 0 {
		return 0
	}
	return len(probs) - 1
}

// Mirrors 4d-engine logits_project + sampling4d flat path (CGO-disabled builds).
func projectStubLogitsGo(last []float32, lifted []float32, vocabSize int, step int, logFirst bool) []float32 {
	if len(last) < 4 || vocabSize <= 0 {
		return nil
	}
	l0, l1, l2, l3 := last[0], last[1], last[2], last[3]
	s := float32(step)
	ll := len(lifted)
	out := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		fi := float32(i)
		wi0 := float32(math.Sin(float64(fi*0.013 + s*0.017)))
		wi1 := float32(math.Cos(float64(fi*0.019 - s*0.011)))
		wi2 := float32(math.Sin(float64(fi*0.023 + s*0.029)))
		wi3 := float32(math.Cos(float64(fi*0.031 - s*0.007)))
		z := l0*wi0 + l1*wi1 + l2*wi2 + l3*wi3
		if ll > 0 {
			z += 0.015 * lifted[i%ll]
		}
		z += 0.001 * float32(math.Mod(float64(fi), 97))
		out[i] = float32(math.Tanh(float64(z)) * 2.0)
	}
	if logFirst && vocabSize >= 5 && step == 0 {
		_, _ = fmt.Fprintf(os.Stderr, "🔧 Logits shape: %d | First 5 logits: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			vocabSize, out[0], out[1], out[2], out[3], out[4])
	}
	return out
}

func sampleNextTokenFlatGo(logits []float32, temperature float32, topK int) int {
	temp := float32(math.Max(float64(temperature), 1e-5))
	scaled := make([]float32, len(logits))
	for i, x := range logits {
		scaled[i] = x / temp
	}
	if topK > 0 && topK < len(scaled) {
		idx := make([]int, len(scaled))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return scaled[idx[a]] > scaled[idx[b]] })
		thr := scaled[idx[topK-1]]
		for i := range scaled {
			if scaled[i] < thr {
				scaled[i] = -1e30
			}
		}
	}
	probs := softmax32(scaled)
	return detSampleIndexGo(probs, logits, temperature, topK)
}

func sampleNextToken4DGo(logits []float32, temperature float32, topK int) uint32 {
	temp := float32(math.Max(float64(temperature), 1e-5))
	v := append([]float32(nil), logits...)
	for len(v)%4 != 0 {
		v = append(v, 0)
	}
	seqLen := len(v) / 4
	if seqLen == 0 {
		return 0
	}
	if seqLen >= 2 {
		smoothed := spacetimeAttentionDispatchStub(v, v, v, seqLen)
		if len(smoothed) == len(v) {
			v = smoothed
		}
	}
	readout := readoutQuat4()
	proj := make([]float32, 0, seqLen)
	for i := 0; i < len(v); i += 4 {
		q := quatNormalize([4]float32{v[i], v[i+1], v[i+2], v[i+3]})
		s := quatDot4(q, readout)*2 + 0.25*(v[i]+v[i+1]+v[i+2]+v[i+3])
		proj = append(proj, s/temp)
	}
	if topK > 0 && topK < len(proj) {
		order := make([]int, len(proj))
		for i := range order {
			order[i] = i
		}
		for a := 0; a < len(order); a++ {
			for b := a + 1; b < len(order); b++ {
				if proj[order[b]] > proj[order[a]] {
					order[a], order[b] = order[b], order[a]
				}
			}
		}
		thr := proj[order[topK-1]]
		for i := range proj {
			if proj[i] < thr {
				proj[i] = -1e30
			}
		}
	}
	probs := softmax32(proj)
	idx := detSampleIndexGo(probs, logits, temperature, topK)
	if idx >= seqLen {
		idx = seqLen - 1
	}
	return uint32(idx)
}

func gemm4dAccumulateCell(a, b []float32, mi, ni, k, n int) float32 {
	k4 := (k / 4) * 4
	var acc float32
	ki := 0
	for ki < k4 {
		qa := quatNormalize([4]float32{
			a[mi*k+ki], a[mi*k+ki+1], a[mi*k+ki+2], a[mi*k+ki+3],
		})
		qb := quatNormalize([4]float32{
			b[ki*n+ni], b[(ki+1)*n+ni], b[(ki+2)*n+ni], b[(ki+3)*n+ni],
		})
		acc += quatDot4(qa, qb)
		ki += 4
	}
	for ki < k {
		acc += a[mi*k+ki] * b[ki*n+ni]
		ki++
	}
	return acc
}

func gemm4dGo(a, b []float32, m, k, n int) []float32 {
	c := make([]float32, m*n)
	for mi := 0; mi < m; mi++ {
		for ni := 0; ni < n; ni++ {
			c[mi*n+ni] = gemm4dAccumulateCell(a, b, mi, ni, k, n)
		}
	}
	return c
}

func gemm4dGoParallel(a, b []float32, m, k, n int) []float32 {
	c := make([]float32, m*n)
	nt := min(8, max(1, runtime.NumCPU()))
	if m < 4 || nt <= 1 {
		return gemm4dGo(a, b, m, k, n)
	}
	var wg sync.WaitGroup
	chunk := (m + nt - 1) / nt
	for t := 0; t < nt; t++ {
		r0 := t * chunk
		r1 := min(r0+chunk, m)
		if r0 >= r1 {
			continue
		}
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			for mi := r0; mi < r1; mi++ {
				for ni := 0; ni < n; ni++ {
					c[mi*n+ni] = gemm4dAccumulateCell(a, b, mi, ni, k, n)
				}
			}
		}(r0, r1)
	}
	wg.Wait()
	return c
}

func (stubEngine) InspectGGUF(path string) ([]byte, error) {
	_ = path
	return nil, fmt.Errorf("four_d_engine: this binary was built with CGO disabled; rebuild with CGO_ENABLED=1 and libfour_d_engine (see Dockerfile)")
}
