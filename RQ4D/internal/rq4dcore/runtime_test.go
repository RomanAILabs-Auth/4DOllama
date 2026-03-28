package rq4dcore

import (
	"context"
	"testing"
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/quantum"
)

func TestRuntimeStartStop(t *testing.T) {
	S := quantum.NewSimulator(2, 2, 2, 2, 2)
	S.InitProductComputational()
	S.NormalizeGlobal(S.ReA, S.ImA)
	eng := NewSharedEngine(S)
	rt := NewRuntime(eng, &Profiler{}, NewArena(4096), nil, RuntimeConfig{
		TickInterval: 10 * time.Millisecond,
		MinIdleWait:  5 * time.Millisecond,
		RingCap:      64,
	})
	ctx, cancel := context.WithCancel(context.Background())
	rt.Start(ctx)
	time.Sleep(40 * time.Millisecond)
	cancel()
	rt.Stop()
}
