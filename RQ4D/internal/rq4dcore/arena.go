// Package rq4dcore implements RQ4D-CORE: unified lattice runtime, scheduler, and safe bridges.
package rq4dcore

// Arena is a bump allocator for short-lived task scratch buffers (user-space only).
// Reset between scheduler epochs to avoid unbounded growth.
type Arena struct {
	buf []byte
	off int
}

// NewArena creates an arena with the given backing capacity in bytes.
func NewArena(cap int) *Arena {
	if cap < 64 {
		cap = 64
	}
	return &Arena{buf: make([]byte, cap)}
}

// Reset discards all allocations.
func (a *Arena) Reset() {
	a.off = 0
}

// Alloc returns a slice of n zeroed bytes from the arena, or nil if OOM.
func (a *Arena) Alloc(n int) []byte {
	if n <= 0 || a == nil {
		return nil
	}
	align := (8 - (a.off & 7)) & 7
	if a.off+align+n > len(a.buf) {
		return nil
	}
	a.off += align
	start := a.off
	a.off += n
	s := a.buf[start:a.off]
	clear(s)
	return s
}
