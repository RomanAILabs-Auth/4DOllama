package ollamareg

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	mediaTypeModel  = "application/vnd.ollama.image.model"
	mediaTypeTensor = "application/vnd.ollama.image.tensor"
)

type registryManifest struct {
	SchemaVersion int    `json:"schemaVersion"`
	MediaType     string `json:"mediaType"`
	Config        struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"config"`
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// PullGGUF downloads the Ollama registry GGUF layer into destDir as "{FileStem}.gguf".
// If ollamaModelsRoot is set and the same blob already exists under blobs/, it is reused (no re-download).
// Progress lines are written to prog (may be nil).
func PullGGUF(ctx context.Context, rawRef, destDir, ollamaModelsRoot string, prog io.Writer) (outPath string, err error) {
	ref, err := ParseRef(rawRef)
	if err != nil {
		return "", err
	}
	_ = os.MkdirAll(destDir, 0o755)

	base := fmt.Sprintf("%s://%s", ref.Scheme, ref.Host)
	cli := &http.Client{Timeout: 0}

	mf, err := fetchManifest(ctx, cli, base, ref, prog)
	if err != nil {
		return "", err
	}

	var best string
	var bestSize int64
	var hasTensor bool
	for _, L := range mf.Layers {
		if L.MediaType == mediaTypeTensor {
			hasTensor = true
		}
		if L.MediaType == mediaTypeModel && L.Digest != "" && L.Size >= bestSize {
			bestSize = L.Size
			best = L.Digest
		}
	}

	if best == "" {
		if hasTensor {
			return "", fmt.Errorf("%w: try `4dollama import-ollama %s` after `ollama pull %s`", ErrNoGGUFLayer, ref.Display(), ref.Display())
		}
		return "", fmt.Errorf("%w (no model blob in manifest)", ErrNoGGUFLayer)
	}

	if ollamaModelsRoot != "" {
		bp := filepath.Join(ollamaModelsRoot, "blobs", strings.ReplaceAll(best, ":", "-"))
		if st, err := os.Stat(bp); err == nil {
			if bestSize <= 0 || st.Size() == bestSize {
				if prog != nil {
					_, _ = fmt.Fprintf(prog, "reusing existing Ollama blob (same digest, no re-download): %s\n", bp)
				}
				return bp, nil
			}
		}
	}

	outPath = filepath.Join(destDir, ref.FileStem()+".gguf")
	if prog != nil {
		_, _ = fmt.Fprintf(prog, "pulling %s → %s\n", ref.Display(), outPath)
	}

	if err := downloadBlobVerified(ctx, cli, base, ref, best, outPath, prog); err != nil {
		return "", err
	}
	return outPath, nil
}

func fetchManifest(ctx context.Context, cli *http.Client, base string, ref Ref, prog io.Writer) (*registryManifest, error) {
	u := strings.TrimSuffix(base, "/") + "/" + ref.ManifestPath()
	if prog != nil {
		_, _ = fmt.Fprintf(prog, "fetching manifest %s\n", u)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", "4dollama/0.2 (+https://github.com/4dollama/4dollama)")

	resp, err := cli.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return nil, ErrUnauthorized
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("manifest %s: %s — %s", u, resp.Status, strings.TrimSpace(string(b)))
	}
	var mf registryManifest
	if err := json.NewDecoder(resp.Body).Decode(&mf); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	return &mf, nil
}

func downloadBlobVerified(ctx context.Context, cli *http.Client, base string, ref Ref, digest, dest string, prog io.Writer) error {
	want, err := parseDigest(digest)
	if err != nil {
		return err
	}
	u := strings.TrimSuffix(base, "/") + "/" + ref.BlobPath(digest)
	if prog != nil {
		_, _ = fmt.Fprintf(prog, "downloading blob %s\n", digest)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "4dollama/0.2 (+https://github.com/4dollama/4dollama)")

	resp, err := cli.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return ErrUnauthorized
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("blob %s: %s — %s", u, resp.Status, strings.TrimSpace(string(b)))
	}

	tmp := dest + ".partial-" + time.Now().Format("150405")
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	h := sha256.New()
	w := io.MultiWriter(f, h)
	n, err := io.Copy(w, resp.Body)
	_ = f.Close()
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if prog != nil {
		_, _ = fmt.Fprintf(prog, "downloaded %d bytes\n", n)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		_ = os.Remove(tmp)
		return fmt.Errorf("digest mismatch: want sha256:%s got sha256:%s", want, got)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func parseDigest(d string) (hexNoPrefix string, err error) {
	d = strings.TrimSpace(d)
	const p = "sha256:"
	if !strings.HasPrefix(strings.ToLower(d), p) {
		return "", fmt.Errorf("unsupported digest %q", d)
	}
	h := d[len(p):]
	if len(h) != 64 {
		return "", fmt.Errorf("bad sha256 digest length in %q", d)
	}
	return h, nil
}
