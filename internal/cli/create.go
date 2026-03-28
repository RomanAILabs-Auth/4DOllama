package cli

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/4dollama/4dollama/internal/config"
	"github.com/4dollama/4dollama/internal/models"
)

// CmdCreate implements `4dollama create <name> -f Modelfile` (Ollama-shaped).
// Supports FROM for .gguf (copy) and .4dai (JSON romanai merge or single copy; binary shards: single FROM only).
func CmdCreate(modelName, modelfilePath string, log *slog.Logger) int {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		fmt.Fprintln(os.Stderr, "usage: 4dollama create <name> -f <Modelfile>")
		return 2
	}
	modelfilePath, _ = filepath.Abs(modelfilePath)

	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama create: cannot get working directory: %v\n", err)
		return 1
	}
	cwd, _ = filepath.Abs(cwd)

	mf, err := ParseModelfileFile(modelfilePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama create: modelfile: %v\n", err)
		return 1
	}
	paths, err := ResolveFromPaths(mf.FromPaths, cwd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "4dollama create: %v\n", err)
		return 1
	}

	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[4DOLLAMA FATAL] Shard not found! Searched absolute path: %s\n", p)
			fmt.Fprintf(os.Stderr, "  Modelfile FROM paths are resolved from your current working directory: %s\n", cwd)
			fmt.Fprintf(os.Stderr, "  Fix the FROM line or run `4dollama create` from the directory that contains the shard files.\n")
			return 1
		}
		if st.IsDir() {
			fmt.Fprintf(os.Stderr, "[4DOLLAMA FATAL] FROM path is a directory, not a file: %s\n", p)
			return 1
		}
	}

	ext0 := strings.ToLower(filepath.Ext(paths[0]))
	allSame := true
	for _, p := range paths {
		if strings.ToLower(filepath.Ext(p)) != ext0 {
			allSame = false
			break
		}
	}
	if !allSame {
		fmt.Fprintln(os.Stderr, "4dollama create: all FROM paths must share the same extension (.gguf or .4dai)")
		return 1
	}

	cfg := config.Load()
	_ = os.MkdirAll(cfg.ModelsDir, 0o755)
	blobsDir := filepath.Join(cfg.ModelsDir, "blobs")
	_ = os.MkdirAll(blobsDir, 0o755)
	for _, src := range paths {
		dst := filepath.Join(blobsDir, modelName+"__"+filepath.Base(src))
		if err := copyFile(src, dst); err != nil {
			fmt.Fprintf(os.Stderr, "4dollama create: copy shard to blobs: %v\n", err)
			return 1
		}
	}

	switch ext0 {
	case ".gguf":
		if len(paths) > 1 {
			fmt.Fprintln(os.Stderr, "4dollama create: multiple FROM .gguf — using first shard only")
		}
		dst := filepath.Join(cfg.ModelsDir, modelName+".gguf")
		if err := copyFile(paths[0], dst); err != nil {
			fmt.Fprintf(os.Stderr, "4dollama create: %v\n", err)
			return 1
		}
		if log != nil {
			log.Info("created gguf model", slog.String("name", modelName), slog.String("path", dst))
		}
	case ".4dai":
		outPath := filepath.Join(cfg.ModelsDir, modelName+".4dai")
		if len(paths) == 1 {
			bin, err := models.IsRomanaiBinary4DAI(paths[0])
			if err != nil {
				fmt.Fprintf(os.Stderr, "4dollama create: %v\n", err)
				return 1
			}
			if bin {
				if err := copyFile(paths[0], outPath); err != nil {
					fmt.Fprintf(os.Stderr, "4dollama create: %v\n", err)
					return 1
				}
			} else {
				if err := copyFile(paths[0], outPath); err != nil {
					fmt.Fprintf(os.Stderr, "4dollama create: %v\n", err)
					return 1
				}
			}
		} else {
			if err := models.MergeRomanaiJSON4DAIFiles(paths, outPath); err != nil {
				fmt.Fprintf(os.Stderr, "4dollama create: merge .4dai: %v\n", err)
				return 1
			}
		}
		if log != nil {
			log.Info("created romanai model", slog.String("name", modelName), slog.String("path", outPath))
		}
	default:
		fmt.Fprintf(os.Stderr, "4dollama create: unsupported FROM extension %q (use .gguf or .4dai)\n", ext0)
		return 1
	}

	sidecar := filepath.Join(cfg.ModelsDir, modelName+".Modelfile")
	if err := copyFile(modelfilePath, sidecar); err != nil {
		fmt.Fprintf(os.Stderr, "4dollama create: sidecar: %v\n", err)
		return 1
	}

	fmt.Printf("success: model %q created under %s\n", modelName, cfg.ModelsDir)
	return 0
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}
