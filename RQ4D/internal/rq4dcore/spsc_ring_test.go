package rq4dcore

import (
	"testing"
)

func TestSPSCRingSequential(t *testing.T) {
	r := newSPSCRing(32)
	for i := 0; i < 20; i++ {
		if !r.push(Task{Kind: TaskLatticeStep, Steps: i}) {
			t.Fatalf("push %d failed", i)
		}
	}
	for i := 0; i < 20; i++ {
		task, ok := r.pop()
		if !ok {
			t.Fatalf("pop %d failed", i)
		}
		if task.Steps != i {
			t.Fatalf("want steps %d got %d", i, task.Steps)
		}
	}
	if _, ok := r.pop(); ok {
		t.Fatal("expected empty")
	}
}

