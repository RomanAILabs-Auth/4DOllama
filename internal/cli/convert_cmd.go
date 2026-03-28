package cli

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/convert"
)

// CmdConvert implements `4dollama convert` — GGUF → native Cl(4,0) F16 .4dai, optional install + run.
func CmdConvert(ggufPath, outPath, modelName string, install, doRun bool, runPrompt string, log *slog.Logger) int {
	ggufPath = strings.TrimSpace(ggufPath)
	if ggufPath == "" {
		fmt.Fprintln(os.Stderr, "usage: 4dollama convert [flags] <path/to/model.gguf> [prompt...]")
		return 2
	}
	absGGUF, err := filepath.Abs(ggufPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama convert: %v\n", err)
		return 1
	}
	if _, err := os.Stat(absGGUF); err != nil {
		fmt.Fprintf(os.Stderr, "4dollama convert: %v\n", err)
		return 1
	}

	cfg := config.Load()
	stem := modelName
	if stem == "" {
		stem = modelStemFromGGUF(absGGUF)
	}

	var finalDest string
	if install {
		_ = os.MkdirAll(cfg.ModelsDir, 0o755)
		finalDest = filepath.Join(cfg.ModelsDir, stem+".4dai")
	}
	workOut := strings.TrimSpace(outPath)
	if workOut == "" && install {
		workOut = finalDest
	}

	prog := convert.NewProgressSink(os.Stderr)
	fmt.Fprintf(os.Stderr, "4dollama convert: lifting GGUF → Cl(4,0) F16 safetensors (.4dai compatible)\n")
	fmt.Fprintf(os.Stderr, "  source: %s\n", absGGUF)
	if workOut != "" {
		fmt.Fprintf(os.Stderr, "  output: %s\n", workOut)
	}

	res, err := convert.Run(convert.Options{
		GGUFPath: absGGUF,
		OutPath:  workOut,
		Progress: prog,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama convert: %v\n", err)
		return 1
	}

	if install {
		outAbs := res.OutPath
		if finalDest != "" && !strings.EqualFold(filepath.Clean(outAbs), filepath.Clean(finalDest)) {
			if err := copyFile(outAbs, finalDest); err != nil {
				fmt.Fprintf(os.Stderr, "4dollama convert: install copy: %v\n", err)
				return 1
			}
			_ = os.Remove(outAbs)
			outAbs = finalDest
		}
		if err := writeConvertedModelfile(cfg.ModelsDir, stem, outAbs, absGGUF); err != nil {
			fmt.Fprintf(os.Stderr, "4dollama convert: modelfile: %v\n", err)
			return 1
		}
		fmt.Fprintf(os.Stderr, "installed model %q → %s\n", stem, outAbs)
	} else {
		fmt.Fprintf(os.Stderr, "wrote %s (pass --install to register under FOURD_MODELS)\n", res.OutPath)
	}

	if doRun {
		if runPrompt == "" {
			runPrompt = "Brief one-line greeting as a 4D-lifted model (RomanAI / Cl(4,0) carrier)."
		}
		fmt.Fprintf(os.Stderr, "4dollama convert: running smoke test (FOURD_INFERENCE=stub recommended)…\n")
		if err := ensureServerRunning(log, true); err != nil {
			fmt.Fprintf(os.Stderr, "4dollama convert: %v\n", err)
			return 1
		}
		return cmdRunGenerate(stem, runPrompt)
	}

	fmt.Fprintf(os.Stderr, "done: %d tensors, arch=%q, %.2f MB on disk\n", res.TensorCount, res.Arch, float64(res.BytesWritten)/(1024*1024))
	fmt.Fprintf(os.Stderr, "try: 4dollama run %s \"your prompt\"\n", stem)
	return 0
}

func modelStemFromGGUF(path string) string {
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	var b strings.Builder
	for _, r := range base {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteRune('_')
		}
	}
	s := strings.Trim(b.String(), "_")
	if s == "" {
		return "model"
	}
	return strings.ToLower(s)
}

func writeConvertedModelfile(modelsDir, stem, fourDaiAbs, sourceGGUF string) error {
	body := fmt.Sprintf("FROM %s\nTOKENIZER_FROM %s\nPARAMETER temperature 0.7\nPARAMETER num_ctx 8192\n",
		filepath.Clean(fourDaiAbs), filepath.Clean(sourceGGUF))
	mf := filepath.Join(modelsDir, stem+".Modelfile")
	return os.WriteFile(mf, []byte(body), 0o644)
}
