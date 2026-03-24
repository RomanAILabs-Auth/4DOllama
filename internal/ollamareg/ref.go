package ollamareg

import (
	"net/url"
	"os"
	"strings"
)

// Ref is a resolved Ollama registry reference (same defaults as github.com/ollama/ollama/types/model).
type Ref struct {
	Scheme    string
	Host      string
	Namespace string
	Model     string
	Tag       string
}

// DefaultRegistryBase returns the Ollama public registry URL (scheme + host).
func DefaultRegistryBase() string {
	e := strings.TrimSpace(os.Getenv("OLLAMA_REGISTRY"))
	if e == "" {
		e = strings.TrimSpace(os.Getenv("FOURD_REGISTRY"))
	}
	if e == "" {
		return "https://registry.ollama.ai"
	}
	if !strings.Contains(e, "://") {
		e = "https://" + e
	}
	return strings.TrimSuffix(e, "/")
}

// ParseRef parses strings like "llama3.2", "llama3.2:latest", "library/llama3.2:8b",
// or "registry.ollama.ai/library/llama3.2:latest".
func ParseRef(raw string) (Ref, error) {
	base := DefaultRegistryBase()
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		u = &url.URL{Scheme: "https", Host: "registry.ollama.ai"}
	}
	scheme := u.Scheme
	if scheme == "" {
		scheme = "https"
	}
	host := u.Host

	r := Ref{
		Scheme:    scheme,
		Host:      host,
		Namespace: "library",
		Tag:       "latest",
	}

	s := strings.TrimSpace(raw)
	if s == "" {
		return r, errEmptyRef
	}

	// Strip optional registry base prefix (https://host/ or host/)
	if strings.HasPrefix(s, "https://") || strings.HasPrefix(s, "http://") {
		pu, err := url.Parse(s)
		if err == nil && pu.Host != "" {
			r.Scheme = pu.Scheme
			if r.Scheme == "" {
				r.Scheme = "https"
			}
			r.Host = pu.Host
			s = strings.TrimPrefix(pu.Path, "/")
		}
	} else if strings.Contains(s, "/") {
		parts := strings.SplitN(s, "/", 2)
		if len(parts) == 2 && strings.Contains(parts[0], ".") && !strings.Contains(parts[0], ":") {
			r.Host = parts[0]
			s = parts[1]
		}
	}

	// Tag: last ':' that appears after the last '/'
	if li := strings.LastIndex(s, ":"); li > strings.LastIndex(s, "/") {
		r.Tag = s[li+1:]
		s = s[:li]
	}

	parts := splitPath(s)
	switch len(parts) {
	case 0:
		return r, errEmptyRef
	case 1:
		r.Model = parts[0]
	case 2:
		r.Namespace, r.Model = parts[0], parts[1]
	default:
		// host/ns/model with extra segments → join tail as model name
		r.Namespace = parts[0]
		r.Model = strings.Join(parts[1:], "/")
	}

	if r.Model == "" {
		return r, errEmptyRef
	}
	return r, nil
}

func splitPath(s string) []string {
	var out []string
	for _, p := range strings.Split(s, "/") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// Display returns a short human-readable name (model:tag).
func (r Ref) Display() string {
	return r.Model + ":" + r.Tag
}

// FileStem is a safe filename stem for a downloaded GGUF.
// `model:latest` → `model.gguf` so `4dollama run model` resolves; other tags stay unique.
func (r Ref) FileStem() string {
	var s string
	if r.Tag == "" || strings.EqualFold(r.Tag, "latest") {
		s = r.Model
	} else {
		s = r.Model + "-" + r.Tag
	}
	s = strings.ReplaceAll(s, string(os.PathSeparator), "_")
	s = strings.ReplaceAll(s, ":", "-")
	return s
}

// ManifestPath returns the registry API path for the manifest (no leading slash).
func (r Ref) ManifestPath() string {
	return "v2/" + r.Namespace + "/" + r.Model + "/manifests/" + r.Tag
}

// BlobPath returns the registry API path for a digest (e.g. sha256:abc...).
func (r Ref) BlobPath(digest string) string {
	return "v2/" + r.Namespace + "/" + r.Model + "/blobs/" + digest
}
