package engine

// Backend describes how inference metadata is obtained.
type Backend string

const (
	BackendNative = Backend("four_d_engine+cgo")
	BackendStub   = Backend("stub")
)

// Info identifies the loaded engine implementation.
type Info struct {
	Version string
	Backend Backend
}

// Engine abstracts GGUF inspection and future token generation.
type Engine interface {
	Info() Info
	// Capabilities returns JSON describing 4D/quaternion/FFI features (see fd4_capabilities_json).
	Capabilities() ([]byte, error)
	// InspectGGUF returns JSON bytes from the native scanner, or an error if unavailable.
	InspectGGUF(path string) ([]byte, error)
	// Compute4DDemo runs quaternion rotation demo (Rust via CGO when native).
	Compute4DDemo(in []float32) ([]float32, error)
	// GGUFParamCount returns total tensor elements from GGUF metadata.
	GGUFParamCount(path string) (int64, error)
	// GGUFSampleLift reads a weight sample, lifts to 4D quaternions (native only).
	GGUFSampleLift(path string, maxSample int) (lifted []float32, paramCount int64, err error)
	// Rope4DSequence applies quaternion RoPE per token quad on prompt embeddings (pad len % 4 == 0).
	Rope4DSequence(in []float32) ([]float32, error)
	// SpacetimeAttention4D runs causal quaternion attention; len(q)==len(k)==len(v) >= seqLen*4.
	SpacetimeAttention4D(q, k, v []float32, seqLen int) ([]float32, error)
	// SampleNextToken4D softmax-samples a quaternion-row index from logits (len multiple of 4 after internal pad).
	SampleNextToken4D(logits []float32, temperature float32, topK int) (uint32, error)
	// ProjectStubLogits maps the final 4D state to per-vocabulary logits (stub autoregressive path).
	ProjectStubLogits(last []float32, lifted []float32, vocabSize int, step int, logFirst bool) ([]float32, error)
	// SampleNextTokenFlat samples a token index from 1D logits (full vocab).
	SampleNextTokenFlat(logits []float32, temperature float32, topK int) (int, error)
	// Gemm4D computes C[m,n] = A[m,k]*B[k,n] (row-major); returns length m*n.
	Gemm4D(a, b []float32, m, k, n int) ([]float32, error)
	// GPUBackend returns "cpu", "cuda", or "metal" (parallel scheduling path when not cpu).
	GPUBackend() string
}

