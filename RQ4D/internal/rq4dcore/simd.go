package rq4dcore

// SIMDInner is a placeholder for future architecture-specific vectorized dot/axpy.
// Hot paths remain in package quantum; profile with -tags and pprof first.
func SIMDInner(a, b []float64) float64 {
	var s float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}
