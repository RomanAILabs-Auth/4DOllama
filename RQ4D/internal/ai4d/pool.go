package ai4d

import "sync"

// FloatSlicePool reuses []float64 backing storage for activations (hot paths avoid repeated heap growth).
type FloatSlicePool struct {
	p sync.Pool
}

func NewFloatSlicePool() *FloatSlicePool {
	return &FloatSlicePool{
		p: sync.Pool{
			New: func() any {
				s := make([]float64, 0, 1024)
				return &s
			},
		},
	}
}

func (fp *FloatSlicePool) Get(n int) []float64 {
	if fp == nil {
		return make([]float64, n)
	}
	sp := fp.p.Get().(*[]float64)
	s := *sp
	if cap(s) < n {
		s = make([]float64, n)
	} else {
		s = s[:n]
		for i := range s {
			s[i] = 0
		}
	}
	return s
}

func (fp *FloatSlicePool) Put(s []float64) {
	if fp == nil || cap(s) == 0 {
		return
	}
	s = s[:0]
	fp.p.Put(&s)
}
