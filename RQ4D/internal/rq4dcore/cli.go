package rq4dcore

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/ai4d"
	"github.com/RomanAILabs-Auth/RomaQuantum4D/internal/quantum"
)

// Main is the entrypoint for rq4d-core and `rq4d core ...`.
func Main(args []string) int {
	fs := flag.NewFlagSet("rq4d-core", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	mode := fs.String("mode", "batch", "daemon | ai-daemon | batch | interactive")
	lx := fs.Int("lx", 8, "lattice x")
	ly := fs.Int("ly", 8, "lattice y")
	lz := fs.Int("lz", 8, "lattice z")
	dim := fs.Int("dim", 2, "local dimension 2|4|8")
	dt := fs.Float64("dt", 0.05, "Trotter delta t")
	steps := fs.Int("steps", 20, "batch: number of Step() calls")
	workers := fs.Int("workers", 0, "worker hint (0 = GOMAXPROCS)")
	jbond := fs.Float64("j", 0.3, "XX bond J")
	hz0 := fs.Float64("hz", 0.2, "uniform Z field")
	hx0 := fs.Float64("hx", 0.15, "uniform X field")
	backend := fs.String("backend", "meanfield", "meanfield | tn | cpu")
	chi := fs.Int("chi", 1, "TN bond dimension chi")
	tickMs := fs.Int("tick-ms", 0, "daemon: automatic step interval ms (0 = off)")
	idleMs := fs.Int("idle-ms", 50, "executor sleep when queues empty")
	ringCap := fs.Int("ring-cap", 256, "scheduler channel / ring capacity (power of 2)")
	bridge := fs.String("bridge", "127.0.0.1:8744", "HTTP bridge listen addr")
	bridgeOpen := fs.Bool("bridge-allow-non-loopback", false, "allow non-loopback bridge bind (opt-in)")
	ollama := fs.String("ollama-url", "", "optional Ollama base URL, enables /v1/ollama/forward")
	pprof := fs.String("pprof", "", "optional pprof listen addr e.g. 127.0.0.1:6060")
	aiBackend := fs.String("ai-backend", "native", "native | python (math backend for romanai.4dai hub)")
	pythonBridge := fs.String("python-bridge-url", "http://127.0.0.1:8777", "base URL for optional PyTorch HTTP worker (/train, /infer)")

	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *lx < 1 || *ly < 1 || *lz < 1 {
		fmt.Fprintln(os.Stderr, "lx, ly, lz must be >= 1")
		return 2
	}
	if *dim != 2 && *dim != 4 && *dim != 8 {
		fmt.Fprintln(os.Stderr, "dim must be 2, 4, or 8")
		return 2
	}
	if *backend != "meanfield" && *backend != "tn" && *backend != "cpu" {
		fmt.Fprintf(os.Stderr, "backend must be meanfield, tn, or cpu\n")
		return 2
	}
	if *mode != "daemon" && *mode != "ai-daemon" && *mode != "batch" && *mode != "interactive" {
		fmt.Fprintf(os.Stderr, "unknown --mode %q\n", *mode)
		return 2
	}
	ab := strings.ToLower(strings.TrimSpace(*aiBackend))
	if ab != "native" && ab != "python" {
		fmt.Fprintf(os.Stderr, "ai-backend must be native or python\n")
		return 2
	}
	if *chi < 1 || *chi > quantum.MaxChiCap {
		fmt.Fprintf(os.Stderr, "chi must be in [1,%d]\n", quantum.MaxChiCap)
		return 2
	}

	w := *workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	S := quantum.NewSimulator(*lx, *ly, *lz, *dim, w)
	S.SetQuantumBackend(*backend, *chi)
	S.Dt = *dt
	S.JBond = *jbond
	S.SetUniformFields(*hz0, *hx0)
	S.Workers = w
	S.InitProductComputational()
	S.NormalizeGlobal(S.ReA, S.ImA)
	if S.Backend == quantum.BackendTN {
		S.SyncAllRhoFromPsi(S.ReA, S.ImA)
		copy(S.RhoB, S.RhoA)
	}

	eng := NewSharedEngine(S)
	prof := &Profiler{}
	arena := NewArena(1 << 20)
	tick := time.Duration(*tickMs) * time.Millisecond
	cfg := RuntimeConfig{
		TickInterval: tick,
		MinIdleWait:  time.Duration(*idleMs) * time.Millisecond,
		RingCap:      *ringCap,
	}
	rt := NewRuntime(eng, prof, arena, NullDevice{}, cfg)
	hub := ai4d.NewHub(ab, *pythonBridge)
	hub.Continuous = *mode == "ai-daemon"
	rt.AI = hub

	switch *mode {
	case "batch":
		return runBatch(rt, *steps)
	case "interactive":
		return runInteractive(rt)
	case "daemon", "ai-daemon":
		return runDaemon(rt, *bridge, *bridgeOpen, *ollama, *pprof)
	default:
		fmt.Fprintf(os.Stderr, "unknown --mode %q\n", *mode)
		return 2
	}
}

func runBatch(rt *Runtime, steps int) int {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rt.Start(ctx)
	defer rt.Stop()
	resp := make(chan TaskResult, 1)
	if !rt.SubmitSim(Task{Kind: TaskLatticeStep, Steps: steps, Resp: resp}) {
		fmt.Fprintln(os.Stderr, "could not enqueue batch step task")
		return 1
	}
	select {
	case r := <-resp:
		if r.Err != nil {
			fmt.Fprintln(os.Stderr, r.Err)
			return 1
		}
		fmt.Printf("global_norm2=%.12g expectation_H_mf=%.12g state_sha256=%s\n", r.Norm2, r.ExpectH, r.StateSHA)
	case <-time.After(2 * time.Minute):
		fmt.Fprintln(os.Stderr, "timeout waiting for batch")
		return 1
	}
	return 0
}

func runInteractive(rt *Runtime) int {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rt.Start(ctx)
	defer rt.Stop()

	fmt.Println("RQ4D-CORE interactive shell. Commands: step [n], measure <site> [seed], diag, transform <text>, r4dline <Roma4D AI stmt>, quit")
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("rq4d-core> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		switch parts[0] {
		case "quit", "exit":
			return 0
		case "diag":
			n2, eh, sha := rt.Eng.SnapshotDiagnostics()
			fmt.Printf("norm2=%.12g H=%.12g sha=%s\n", n2, eh, sha)
		case "step":
			n := 1
			if len(parts) > 1 {
				if v, err := strconv.Atoi(parts[1]); err == nil && v > 0 {
					n = v
				}
			}
			resp := make(chan TaskResult, 1)
			if !rt.SubmitSim(Task{Kind: TaskLatticeStep, Steps: n, Resp: resp}) {
				fmt.Println("queue full")
				continue
			}
			r := <-resp
			fmt.Printf("norm2=%.12g H=%.12g sha=%s\n", r.Norm2, r.ExpectH, r.StateSHA)
		case "measure":
			if len(parts) < 2 {
				fmt.Println("usage: measure <site> [seed]")
				continue
			}
			site, _ := strconv.Atoi(parts[1])
			seed := int64(0)
			if len(parts) > 2 {
				seed, _ = strconv.ParseInt(parts[2], 10, 64)
			}
			resp := make(chan TaskResult, 1)
			if !rt.SubmitSim(Task{Kind: TaskMeasurementSample, Site: site, Seed: seed, Resp: resp}) {
				fmt.Println("queue full")
				continue
			}
			r := <-resp
			fmt.Printf("outcome=%d norm2=%.12g sha=%s\n", r.MeasureK, r.Norm2, r.StateSHA)
		case "transform":
			prompt := strings.TrimPrefix(line, "transform")
			prompt = strings.TrimSpace(prompt)
			resp := make(chan TaskResult, 1)
			if !rt.SubmitExternal(Task{Kind: TaskExternalRequest, Prompt: prompt, Resp: resp}) {
				fmt.Println("external queue full")
				continue
			}
			r := <-resp
			fmt.Println(r.Transform)
		case "r4dline":
			stmt := strings.TrimSpace(strings.TrimPrefix(line, "r4dline"))
			if stmt == "" {
				fmt.Println("usage: r4dline MODEL LOAD \"m.4dai\"")
				continue
			}
			cmd, err := ai4d.ParseLine(stmt)
			if err != nil {
				fmt.Println(err)
				continue
			}
			t, tOK := TaskFromAICommand(cmd)
			if !tOK {
				continue
			}
			resp := make(chan TaskResult, 1)
			t.Resp = resp
			if !rt.SubmitAI(t) {
				fmt.Println("ai queue full or hub disabled")
				continue
			}
			r := <-resp
			if r.Err != nil {
				fmt.Println("error:", r.Err)
				continue
			}
			if r.AIOut != nil {
				fmt.Printf("infer_out=%v\n", r.AIOut)
			} else {
				fmt.Println("ok")
			}
		default:
			fmt.Println("unknown command")
		}
	}
	return 0
}

func runDaemon(rt *Runtime, bridgeAddr string, bridgeOpen bool, ollamaURL, pprofAddr string) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var pp *http.Server
	if pprofAddr != "" {
		pp = StartPprof(pprofAddr)
	}

	rt.Start(ctx)

	br, err := NewBridgeServer(rt, profFromRuntime(rt), bridgeAddr, bridgeOpen, ollamaURL)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		rt.Stop()
		return 1
	}
	go func() {
		if err := br.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "bridge: %v\n", err)
		}
	}()

	modeLabel := "daemon"
	if rt.AI != nil && rt.AI.Continuous {
		modeLabel = "ai-daemon"
	}
	fmt.Printf("RQ4D-CORE %s (user-space). Bridge http://%s pprof=%q ai-backend=%s\n", modeLabel, bridgeAddr, pprofAddr, rt.AI.Backend)
	<-ctx.Done()
	shctx, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	_ = br.Shutdown(shctx)
	rt.Stop()
	if pp != nil {
		_ = pp.Shutdown(shctx)
	}
	fmt.Println("shutdown complete")
	return 0
}

func profFromRuntime(rt *Runtime) *Profiler {
	if rt == nil {
		return nil
	}
	return rt.Prof
}
