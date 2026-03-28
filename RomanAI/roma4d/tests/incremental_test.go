package tests

import (
	"path/filepath"
	"testing"

	"github.com/RomanAILabs-Auth/Roma4D/src/compiler"
)

func TestPackageInputHashStableShape(t *testing.T) {
	root := roma4dRoot(t)
	entry := filepath.Join("examples", "hello_4d.r4d")
	h1, err := compiler.PackageInputHash(root, entry)
	if err != nil {
		t.Fatal(err)
	}
	if len(h1) != 64 {
		t.Fatalf("expected 64-char hex sha256, got len=%d", len(h1))
	}
	h2, err := compiler.PackageInputHash(root, entry)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Fatalf("same inputs should hash identically: %q vs %q", h1, h2)
	}
	cache := compiler.IncrementalCacheDir(root)
	if filepath.Base(cache) != "cache" {
		t.Fatalf("unexpected cache dir: %s", cache)
	}
}
