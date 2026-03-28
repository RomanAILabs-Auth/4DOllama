package compiler

// Backend selects LLVM, CUDA, or experimental targets.
type Backend int

const (
	BackendLLVM Backend = iota
	BackendCUDA
)

// EmitPlaceholder is a stub for IR emission.
func EmitPlaceholder(backend Backend, _ string) error {
	_ = backend
	return nil
}
