package ollamareg

import "testing"

func TestParseRef_simple(t *testing.T) {
	r, err := ParseRef("llama3.2")
	if err != nil {
		t.Fatal(err)
	}
	if r.Model != "llama3.2" || r.Tag != "latest" || r.Namespace != "library" {
		t.Fatalf("%+v", r)
	}
	if r.FileStem() != "llama3.2" {
		t.Fatalf("stem %q", r.FileStem())
	}
}

func TestParseRef_tag(t *testing.T) {
	r, err := ParseRef("phi3:mini")
	if err != nil {
		t.Fatal(err)
	}
	if r.Model != "phi3" || r.Tag != "mini" {
		t.Fatalf("%+v", r)
	}
	if r.FileStem() != "phi3-mini" {
		t.Fatalf("stem %q", r.FileStem())
	}
}
