package rq4dcore

// Device is an optional accelerator hook (stub). No drivers or kernel mode.
type Device interface {
	Name() string
	// Enqueue is reserved for future GPU/tensor offload of local site ops.
	Enqueue(op string, payload []float64) error
}

// NullDevice is the default CPU-only backend.
type NullDevice struct{}

func (NullDevice) Name() string { return "cpu-null" }

func (NullDevice) Enqueue(op string, payload []float64) error { return nil }
