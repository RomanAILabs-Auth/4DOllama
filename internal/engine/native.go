//go:build cgo

package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../4d-engine/include
#cgo windows LDFLAGS: ${SRCDIR}/../../4d-engine/target/release/four_d_engine.lib
#cgo !windows LDFLAGS: -L${SRCDIR}/../../4d-engine/target/release -lfour_d_engine -lm
#include "four_d_engine.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/4dollama/4dollama/internal/version"
)

type nativeEngine struct{}

// New constructs the cgo-backed four_d_engine.
func New() Engine {
	return nativeEngine{}
}

func (nativeEngine) Info() Info {
	return Info{Version: version.Version, Backend: BackendNative}
}

func (nativeEngine) Capabilities() ([]byte, error) {
	p := C.fd4_capabilities_json()
	if p == nil {
		msg := C.GoString(C.fd4_last_error())
		if msg == "" {
			msg = "fd4_capabilities_json failed"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	defer C.fd4_free_string(p)
	return []byte(C.GoString(p)), nil
}

func (nativeEngine) GGUFParamCount(path string) (int64, error) {
	cs := C.CString(path)
	defer C.free(unsafe.Pointer(cs))
	var n C.uint64_t
	rc := C.fd4_gguf_param_count(cs, &n)
	if rc != 0 {
		return 0, fmt.Errorf("%s", C.GoString(C.fd4_last_error()))
	}
	return int64(n), nil
}

func (nativeEngine) GGUFSampleLift(path string, maxSample int) ([]float32, int64, error) {
	if maxSample <= 0 {
		maxSample = 8192
	}
	cs := C.CString(path)
	defer C.free(unsafe.Pointer(cs))
	out := make([]float32, maxSample+4096)
	var nw C.size_t
	var np C.uint64_t
	rc := C.fd4_gguf_sample_lift(
		cs,
		C.size_t(maxSample),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.size_t(len(out)),
		&nw,
		&np,
	)
	if rc != 0 {
		return nil, 0, fmt.Errorf("%s", C.GoString(C.fd4_last_error()))
	}
	return out[:int(nw)], int64(np), nil
}

func (nativeEngine) Compute4DDemo(in []float32) ([]float32, error) {
	if len(in) == 0 {
		return nil, nil
	}
	pad := (len(in) + 2) / 3 * 3
	out := make([]float32, pad)
	var n C.size_t
	rc := C.fd4_compute_demo(
		(*C.float)(unsafe.Pointer(&in[0])),
		C.size_t(len(in)),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.size_t(len(out)),
		&n,
	)
	if rc != 0 {
		msg := C.GoString(C.fd4_last_error())
		if msg == "" {
			msg = "fd4_compute_demo failed"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	return out[:int(n)], nil
}

func (nativeEngine) Rope4DSequence(in []float32) ([]float32, error) {
	if len(in) == 0 {
		return nil, nil
	}
	pad := (len(in) + 3) / 4 * 4
	out := make([]float32, pad)
	var n C.size_t
	rc := C.fd4_rope4d_sequence(
		(*C.float)(unsafe.Pointer(&in[0])),
		C.size_t(len(in)),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.size_t(len(out)),
		&n,
	)
	if rc != 0 {
		msg := C.GoString(C.fd4_last_error())
		if msg == "" {
			msg = "fd4_rope4d_sequence failed"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	return out[:int(n)], nil
}

func (nativeEngine) SpacetimeAttention4D(q, k, v []float32, seqLen int) ([]float32, error) {
	if seqLen <= 0 || len(q) < seqLen*4 || len(k) < seqLen*4 || len(v) < seqLen*4 {
		return nil, fmt.Errorf("SpacetimeAttention4D: invalid buffers or seqLen")
	}
	need := seqLen * 4
	out := make([]float32, need)
	var n C.size_t
	rc := C.fd4_spacetime_attention(
		(*C.float)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&k[0])),
		(*C.float)(unsafe.Pointer(&v[0])),
		C.size_t(need),
		C.size_t(seqLen),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.size_t(len(out)),
		&n,
	)
	if rc != 0 {
		msg := C.GoString(C.fd4_last_error())
		if msg == "" {
			msg = "fd4_spacetime_attention failed"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	return out[:int(n)], nil
}

func (nativeEngine) SampleNextToken4D(logits []float32, temperature float32, topK int) (uint32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("SampleNextToken4D: empty logits")
	}
	if topK < 0 {
		topK = 0
	}
	if topK > 1<<20 {
		topK = 1 << 20
	}
	id := C.fd4_sample_next_token_4d(
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.size_t(len(logits)),
		C.float(temperature),
		C.size_t(topK),
	)
	return uint32(id), nil
}

func (nativeEngine) ProjectStubLogits(last []float32, lifted []float32, vocabSize int, step int, logFirst bool) ([]float32, error) {
	if len(last) < 4 || vocabSize <= 0 {
		return nil, fmt.Errorf("ProjectStubLogits: invalid args")
	}
	out := make([]float32, vocabSize)
	var n C.size_t
	var liftedPtr *C.float
	var liftedLen C.size_t
	if len(lifted) > 0 {
		liftedPtr = (*C.float)(unsafe.Pointer(&lifted[0]))
		liftedLen = C.size_t(len(lifted))
	}
	lf := C.int(0)
	if logFirst {
		lf = 1
	}
	rc := C.fd4_project_logits_stub(
		(*C.float)(unsafe.Pointer(&last[0])), C.size_t(4),
		liftedPtr, liftedLen,
		C.uint32_t(vocabSize),
		(*C.float)(unsafe.Pointer(&out[0])), C.size_t(len(out)),
		&n,
		C.uint32_t(step),
		lf,
	)
	if rc != 0 {
		return nil, fmt.Errorf("%s", C.GoString(C.fd4_last_error()))
	}
	return out[:int(n)], nil
}

func (nativeEngine) SampleNextTokenFlat(logits []float32, temperature float32, topK int) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("SampleNextTokenFlat: empty logits")
	}
	if topK < 0 {
		topK = 0
	}
	id := C.fd4_sample_next_token_flat(
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.size_t(len(logits)),
		C.float(temperature),
		C.size_t(topK),
	)
	return int(id), nil
}

func (nativeEngine) Gemm4D(a, b []float32, m, k, n int) ([]float32, error) {
	if m <= 0 || k <= 0 || n <= 0 {
		return nil, fmt.Errorf("Gemm4D: invalid dims")
	}
	needA, needB := m*k, k*n
	if len(a) < needA || len(b) < needB {
		return nil, fmt.Errorf("Gemm4D: buffer too small")
	}
	c := make([]float32, m*n)
	rc := C.fd4_gemm4d(
		(*C.float)(unsafe.Pointer(&a[0])),
		C.size_t(len(a)),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.size_t(len(b)),
		(*C.float)(unsafe.Pointer(&c[0])),
		C.size_t(len(c)),
		C.size_t(m),
		C.size_t(k),
		C.size_t(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("%s", C.GoString(C.fd4_last_error()))
	}
	return c, nil
}

func (nativeEngine) GPUBackend() string {
	return C.GoString(C.fd4_gpu_backend_name())
}

func (nativeEngine) InspectGGUF(path string) ([]byte, error) {
	cs := C.CString(path)
	defer C.free(unsafe.Pointer(cs))
	p := C.fd4_gguf_inspect_json(cs)
	if p == nil {
		msg := C.GoString(C.fd4_last_error())
		if msg == "" {
			msg = "fd4_gguf_inspect_json failed"
		}
		return nil, fmt.Errorf("%s", msg)
	}
	defer C.fd4_free_string(p)
	return []byte(C.GoString(p)), nil
}
