package convert

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
)

// Options configures GGUF → Clifford-lifted safetensors (.4dai) conversion.
type Options struct {
	GGUFPath string
	// OutPath is the output file; if empty, defaults to <stem>_4d.4dai next to the GGUF.
	OutPath string
	Progress *ProgressSink
}

// Result summarizes a successful conversion.
type Result struct {
	OutPath      string
	TensorCount  int
	Arch         string
	BytesWritten int64
}

// Run converts one GGUF file into a native safetensors .4dai (F16, Cl(4,0) block layout).
// One tensor is dequantized and packed at a time to limit RAM use on consumer hardware.
func Run(opt Options) (*Result, error) {
	if strings.TrimSpace(opt.GGUFPath) == "" {
		return nil, fmt.Errorf("gguf path required")
	}
	inAbs, err := filepath.Abs(opt.GGUFPath)
	if err != nil {
		return nil, err
	}
	layout, err := ScanGGUF(inAbs)
	if err != nil {
		return nil, fmt.Errorf("gguf scan: %w", err)
	}

	outPath := opt.OutPath
	if outPath == "" {
		base := strings.TrimSuffix(filepath.Base(inAbs), filepath.Ext(inAbs))
		outPath = filepath.Join(filepath.Dir(inAbs), base+"_4d.4dai")
	}
	outAbs, err := filepath.Abs(outPath)
	if err != nil {
		return nil, err
	}

	type job struct {
		safe string
		t    GGUFTensor
	}
	var jobs []job
	dup := make(map[string]int)
	for _, t := range layout.Tensors {
		if t.NElems <= 0 {
			continue
		}
		if _, err := ggmlRowSize(t.GGMLType, t.NElems); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
		}
		base := SanitizeTensorKey(t.Name)
		nDup := dup[base]
		dup[base]++
		safe := base
		if nDup > 0 {
			safe = fmt.Sprintf("%s__%d", base, nDup)
		}
		jobs = append(jobs, job{safe: safe, t: t})
	}
	if len(jobs) == 0 {
		return nil, fmt.Errorf("no non-empty tensors in GGUF")
	}

	neByName := make(map[string]int64, len(jobs))
	names := make([]string, 0, len(jobs))
	tensorBySafe := make(map[string]GGUFTensor, len(jobs))
	for _, j := range jobs {
		neByName[j.safe] = j.t.NElems
		names = append(names, j.safe)
		tensorBySafe[j.safe] = j.t
	}

	ggufFile, err := os.Open(inAbs)
	if err != nil {
		return nil, err
	}
	defer ggufFile.Close()

	var bytesRead int64
	var seq atomic.Uint32
	n := len(names)
	produce := func(safeName string) ([]byte, []int64, error) {
		t := tensorBySafe[safeName]
		rowBytes, err := ggmlRowSize(t.GGMLType, t.NElems)
		if err != nil {
			return nil, nil, err
		}
		raw := make([]byte, rowBytes)
		off := int64(layout.DataBase) + int64(t.Offset)
		if _, err := ggufFile.Seek(off, 0); err != nil {
			return nil, nil, err
		}
		if _, err := io.ReadFull(ggufFile, raw); err != nil {
			return nil, nil, fmt.Errorf("read tensor data: %w", err)
		}
		bytesRead += rowBytes
		f32, err := DequantizeGGML(raw, t.GGMLType, t.NElems)
		if err != nil {
			return nil, nil, err
		}
		data, shape := PackCliffordF16(f32)
		i := int(seq.Load())
		seq.Add(1)
		if opt.Progress != nil {
			opt.Progress.ReportTensor("convert", i, n, t.Name, bytesRead)
		}
		return data, shape, nil
	}

	if err := WriteSafetensorsCliffordF16Stream(outAbs, names, neByName, produce); err != nil {
		return nil, err
	}
	st, _ := os.Stat(outAbs)
	if opt.Progress != nil {
		opt.Progress.Done(fmt.Sprintf("wrote %s (%d tensors, arch=%s)", outAbs, len(jobs), layout.Architecture))
	}
	return &Result{
		OutPath:      outAbs,
		TensorCount:  len(jobs),
		Arch:         layout.Architecture,
		BytesWritten: st.Size(),
	}, nil
}
