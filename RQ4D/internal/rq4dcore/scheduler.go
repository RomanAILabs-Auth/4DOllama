package rq4dcore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/ai4d"
	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/quantum"
)

// RuntimeConfig controls the daemon / batch executor (all user-space).
type RuntimeConfig struct {
	TickInterval time.Duration // 0 = no automatic lattice steps
	MinIdleWait  time.Duration // when queues empty, blocks up to this in the executor
	RingCap      int           // power of two, per SPSC ring
}

// Runtime owns the shared engine, profiler, optional arena, and scheduler rings.
type Runtime struct {
	Eng   *SharedEngine
	Prof  *Profiler
	Arena *Arena
	Dev   Device
	AI    *ai4d.Hub // optional romanai.4dai + Roma4D dispatch target

	cfg       RuntimeConfig
	simRing   *spscRing
	aiRing    *spscRing
	extRing   *spscRing
	simIn     chan Task // merged by coordinator → simRing (single producer)
	aiIn      chan Task
	extIn     chan Task
	coordStop context.CancelFunc
	execStop  context.CancelFunc
	coordWG   sync.WaitGroup
	execWG    sync.WaitGroup
}

// NewRuntime builds rings and channels. Dev may be nil (NullDevice used at call sites).
func NewRuntime(eng *SharedEngine, prof *Profiler, arena *Arena, dev Device, cfg RuntimeConfig) *Runtime {
	if cfg.RingCap < 16 {
		cfg.RingCap = 256
	}
	// SPSC ring size must be power of two.
	n := cfg.RingCap
	p := 16
	for p < n {
		p <<= 1
	}
	cfg.RingCap = p
	if cfg.MinIdleWait <= 0 {
		cfg.MinIdleWait = 50 * time.Millisecond
	}
	if dev == nil {
		dev = NullDevice{}
	}
	return &Runtime{
		Eng:     eng,
		Prof:    prof,
		Arena:   arena,
		Dev:     dev,
		cfg:     cfg,
		simRing: newSPSCRing(cfg.RingCap),
		aiRing:  newSPSCRing(cfg.RingCap),
		extRing: newSPSCRing(cfg.RingCap),
		simIn:   make(chan Task, cfg.RingCap),
		aiIn:    make(chan Task, cfg.RingCap),
		extIn:   make(chan Task, cfg.RingCap),
	}
}

// SubmitSim enqueues simulation-tier work (same priority as tick-driven steps).
func (rt *Runtime) SubmitSim(t Task) bool {
	select {
	case rt.simIn <- t:
		return true
	default:
		if rt.Prof != nil {
			rt.Prof.QueueDropsSim.Add(1)
		}
		return false
	}
}

// SubmitExternal queues work for the external priority tier (below simulation).
func (rt *Runtime) SubmitExternal(t Task) bool {
	select {
	case rt.extIn <- t:
		return true
	default:
		if rt.Prof != nil {
			rt.Prof.QueueDropsExt.Add(1)
		}
		return false
	}
}

// SubmitAI enqueues AI-tier work (priority below simulation, above external).
func (rt *Runtime) SubmitAI(t Task) bool {
	if rt.AI == nil {
		return false
	}
	select {
	case rt.aiIn <- t:
		rt.AI.SchedulerAIQ.Add(1)
		if rt.Prof != nil {
			rt.Prof.AIEnqueued.Add(1)
		}
		return true
	default:
		if rt.Prof != nil {
			rt.Prof.QueueDropsAI.Add(1)
		}
		return false
	}
}

// Start launches coordinator (tick → sim ring) and executor (drain sim then ext).
func (rt *Runtime) Start(ctx context.Context) {
	ctxCoord, cancelC := context.WithCancel(ctx)
	rt.coordStop = cancelC
	ctxExec, cancelE := context.WithCancel(ctx)
	rt.execStop = cancelE

	rt.execWG.Add(1)
	go rt.runExecutor(ctxExec)

	rt.coordWG.Add(1)
	go rt.runCoordinator(ctxCoord)
}

// Stop shuts down coordinator and executor.
func (rt *Runtime) Stop() {
	if rt.coordStop != nil {
		rt.coordStop()
	}
	if rt.execStop != nil {
		rt.execStop()
	}
	rt.coordWG.Wait()
	rt.execWG.Wait()
}

func (rt *Runtime) runCoordinator(ctx context.Context) {
	defer rt.coordWG.Done()
	var tick *time.Ticker
	var tickCh <-chan time.Time
	if rt.cfg.TickInterval > 0 {
		tick = time.NewTicker(rt.cfg.TickInterval)
		defer tick.Stop()
		tickCh = tick.C
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-tickCh:
			t := Task{Kind: TaskLatticeStep, Steps: 1}
			if !rt.simRing.push(t) && rt.Prof != nil {
				rt.Prof.QueueDropsSim.Add(1)
			}
		case t := <-rt.simIn:
			if !rt.simRing.push(t) && rt.Prof != nil {
				rt.Prof.QueueDropsSim.Add(1)
			}
		case t := <-rt.aiIn:
			if !rt.aiRing.push(t) && rt.Prof != nil {
				rt.Prof.QueueDropsAI.Add(1)
			}
		case t := <-rt.extIn:
			if !rt.extRing.push(t) && rt.Prof != nil {
				rt.Prof.QueueDropsExt.Add(1)
			}
		}
	}
}

func (rt *Runtime) runExecutor(ctx context.Context) {
	defer rt.execWG.Done()
	var idleStart time.Time
	for {
		if ctx.Err() != nil {
			return
		}
		drained := false
		for {
			t, ok := rt.simRing.pop()
			if !ok {
				break
			}
			rt.execTask(t)
			drained = true
		}
		for {
			t, ok := rt.aiRing.pop()
			if !ok {
				break
			}
			rt.execTask(t)
			drained = true
		}
		if t, ok := rt.extRing.pop(); ok {
			rt.execTask(t)
			drained = true
		}
		if !drained {
			if rt.Prof != nil {
				if idleStart.IsZero() {
					idleStart = time.Now()
				} else {
					rt.Prof.ExecutorIdleNanos.Add(uint64(time.Since(idleStart).Nanoseconds()))
					idleStart = time.Now()
				}
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(rt.cfg.MinIdleWait):
			}
		} else {
			idleStart = time.Time{}
		}
	}
}

func (rt *Runtime) execTask(t Task) {
	switch t.Kind {
	case TaskLatticeStep:
		n := t.Steps
		if n < 1 {
			n = 1
		}
		rt.Eng.WithWrite(func(S *quantum.Simulator) {
			for i := 0; i < n; i++ {
				S.Step()
			}
		})
		if rt.Prof != nil {
			rt.Prof.LatticeSteps.Add(uint64(n))
		}
		if t.Resp != nil {
			norm2, expectH, sha := rt.Eng.SnapshotDiagnostics()
			select {
			case t.Resp <- TaskResult{Norm2: norm2, ExpectH: expectH, StateSHA: sha}:
			default:
			}
		}
	case TaskBondUpdate:
		rt.Eng.WithWrite(func(S *quantum.Simulator) { S.Step() })
		if rt.Prof != nil {
			rt.Prof.BondTasks.Add(1)
		}
		if t.Resp != nil {
			norm2, expectH, sha := rt.Eng.SnapshotDiagnostics()
			select {
			case t.Resp <- TaskResult{Norm2: norm2, ExpectH: expectH, StateSHA: sha}:
			default:
			}
		}
	case TaskSchmidtTruncate:
		if rt.Prof != nil {
			rt.Prof.SchmidtTasks.Add(1)
		}
		if t.Resp != nil {
			norm2, expectH, sha := rt.Eng.SnapshotDiagnostics()
			select {
			case t.Resp <- TaskResult{Norm2: norm2, ExpectH: expectH, StateSHA: sha}:
			default:
			}
		}
	case TaskMeasurementSample:
		seed := t.Seed
		if seed == 0 {
			seed = 42
		}
		site := t.Site
		var k int
		rt.Eng.WithWrite(func(S *quantum.Simulator) {
			rng := quantum.NewRNG(seed)
			k = S.MeasureSite(site, rng, true)
		})
		if rt.Prof != nil {
			rt.Prof.Measurements.Add(1)
		}
		if t.Resp != nil {
			norm2, expectH, sha := rt.Eng.SnapshotDiagnostics()
			select {
			case t.Resp <- TaskResult{MeasureK: k, Norm2: norm2, ExpectH: expectH, StateSHA: sha}:
			default:
			}
		}
	case TaskExternalRequest:
		if rt.Prof != nil {
			rt.Prof.ExternalHandled.Add(1)
		}
		if t.WarmupSteps > 0 {
			w := t.WarmupSteps
			rt.Eng.WithWrite(func(S *quantum.Simulator) {
				for i := 0; i < w; i++ {
					S.Step()
				}
			})
			if rt.Prof != nil {
				rt.Prof.LatticeSteps.Add(uint64(w))
			}
		}
		norm2, expectH, sha := rt.Eng.SnapshotDiagnostics()
		tr := TaskResult{
			Norm2:     norm2,
			ExpectH:   expectH,
			StateSHA:  sha,
			Transform: PhysicsTransform(norm2, sha, t.Prompt),
		}
		if t.Resp != nil {
			select {
			case t.Resp <- tr:
			default:
			}
		}
	case TaskAILoadModel:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		rt.aiTaskReply(t, rt.AI.Load(t.AIPath))
	case TaskAISaveModel:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		rt.aiTaskReply(t, rt.AI.Save(t.AIPath))
	case TaskAILayerClifford:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		rt.aiTaskReply(t, rt.AI.InitCliffordLayer(t.AICliffN))
	case TaskAIRotate4D:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		rt.aiTaskReply(t, rt.AI.Rotate(t.AIPlane, t.AIAngle))
	case TaskAITrainStep:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		st := t.Steps
		if st < 1 {
			st = 1
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		err := rt.AI.Train(ctx, st, t.AILR)
		cancel()
		rt.aiTaskReply(t, err)
	case TaskAIInference:
		if rt.AI == nil {
			rt.aiTaskReplyOut(t, nil, fmt.Errorf("ai hub disabled"))
			break
		}
		var inp []float64
		if t.AIBody != "" {
			_ = json.Unmarshal([]byte(t.AIBody), &inp)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		out, err := rt.AI.Infer(ctx, inp)
		cancel()
		rt.aiTaskReplyOut(t, out, err)
	case TaskAIEvolveField:
		if rt.AI == nil {
			rt.aiTaskReply(t, fmt.Errorf("ai hub disabled"))
			break
		}
		if t.AIField != "" && t.AIField != "quantum" {
			rt.aiTaskReply(t, fmt.Errorf("unknown EVOLVE field %q", t.AIField))
			break
		}
		st := t.Steps
		if st < 1 {
			st = 1
		}
		rt.aiTaskReply(t, rt.AI.EvolveQuantum(st))
	}
	_ = rt.Dev
}

func (rt *Runtime) aiTaskReply(t Task, err error) {
	if rt.Prof != nil {
		rt.Prof.AICompleted.Add(1)
	}
	if t.Resp != nil {
		select {
		case t.Resp <- TaskResult{Err: err}:
		default:
		}
	}
}

func (rt *Runtime) aiTaskReplyOut(t Task, out []float64, err error) {
	if rt.Prof != nil {
		rt.Prof.AICompleted.Add(1)
	}
	if t.Resp != nil {
		var copyOut []float64
		if out != nil {
			copyOut = append([]float64(nil), out...)
		}
		select {
		case t.Resp <- TaskResult{Err: err, AIOut: copyOut}:
		default:
		}
	}
}
