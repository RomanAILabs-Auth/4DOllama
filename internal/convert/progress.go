package convert

import (
	"fmt"
	"io"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/term"
)

// ProgressSink receives conversion progress (thread-safe).
type ProgressSink struct {
	w       io.Writer
	termW   int
	start   time.Time
	lastPct atomic.Int32
}

func NewProgressSink(w io.Writer) *ProgressSink {
	p := &ProgressSink{w: w, start: time.Now()}
	if w == nil {
		return p
	}
	if tw, _, err := term.GetSize(int(sinkFd(w))); err == nil && tw > 20 {
		p.termW = tw
	}
	return p
}

func sinkFd(w io.Writer) uintptr {
	type fd interface{ Fd() uintptr }
	if f, ok := w.(fd); ok {
		return f.Fd()
	}
	return 0
}

// ReportTensor updates the line: phase, index/total, optional name, bytes done, rate.
func (p *ProgressSink) ReportTensor(phase string, i, n int, name string, bytesDone int64) {
	if p == nil || p.w == nil {
		return
	}
	elapsed := time.Since(p.start).Seconds()
	var rate float64
	if elapsed > 0.01 {
		rate = float64(bytesDone) / elapsed
	}
	pct := int(0)
	if n > 0 {
		pct = (100 * (i + 1)) / n
	}
	if int32(pct) == p.lastPct.Load() && i+1 < n {
		return
	}
	p.lastPct.Store(int32(pct))

	barW := 28
	if p.termW > 60 {
		barW = p.termW / 3
		if barW > 40 {
			barW = 40
		}
	}
	filled := pct * barW / 100
	if filled > barW {
		filled = barW
	}
	bar := strings.Repeat("=", filled) + strings.Repeat(" ", barW-filled)

	short := name
	if len(short) > 36 {
		short = "…" + short[len(short)-33:]
	}
	line := fmt.Sprintf("\r%s [%s] %3d%%  %d/%d  %s  %s/s   ",
		phase, bar, pct, i+1, n, short, humanRate(rate))
	fmt.Fprint(p.w, line)
}

func (p *ProgressSink) Done(msg string) {
	if p == nil || p.w == nil {
		return
	}
	fmt.Fprintf(p.w, "\r%s\n", strings.TrimRight(msg, "\n"))
}

func humanRate(bps float64) string {
	switch {
	case bps >= 1e9:
		return fmt.Sprintf("%.2f GB", bps/1e9)
	case bps >= 1e6:
		return fmt.Sprintf("%.2f MB", bps/1e6)
	case bps >= 1e3:
		return fmt.Sprintf("%.1f KB", bps/1e3)
	default:
		return fmt.Sprintf("%.0f B", bps)
	}
}
