package models

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestExpandRomanaiV2PartShards_singlePart(t *testing.T) {
	dir := t.TempDir()
	p1 := filepath.Join(dir, "romanai_v2_part1.4dai")
	if err := os.WriteFile(p1, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := ExpandRomanaiV2PartShards(p1)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0] != p1 {
		t.Fatalf("got %v", out)
	}
}

func TestExpandRomanaiV2PartShards_fourParts(t *testing.T) {
	dir := t.TempDir()
	for i := 1; i <= 4; i++ {
		p := filepath.Join(dir, "romanai_v2_part"+strconv.Itoa(i)+".4dai")
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	p1 := filepath.Join(dir, "romanai_v2_part1.4dai")
	out, err := ExpandRomanaiV2PartShards(p1)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("want 4 shards, got %d", len(out))
	}
}

func TestExpandRomanaiV2PartShards_gapErrors(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"romanai_v2_part1.4dai", "romanai_v2_part2.4dai"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	p1 := filepath.Join(dir, "romanai_v2_part1.4dai")
	_, err := ExpandRomanaiV2PartShards(p1)
	if err == nil {
		t.Fatal("expected error for 2/4 shards")
	}
}
