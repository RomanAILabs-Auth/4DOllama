package ai

import (
	"os"
	"strings"
	"testing"
)

func TestExtractErrorLine(t *testing.T) {
	if n := ExtractErrorLine(`C:\tmp\x.r4d:12:3: oops`); n != 12 {
		t.Fatalf("got %d", n)
	}
	if n := ExtractErrorLine("no line here"); n != 0 {
		t.Fatalf("got %d", n)
	}
}

func TestRunExpertMode_strictPassthrough(t *testing.T) {
	fixed, msg := RunExpertMode("/x.r4d", "raw", 0, false)
	if fixed || msg != "raw" {
		t.Fatalf("strict: fixed=%v msg=%q", fixed, msg)
	}
}

func TestRunExpertMode_importStar(t *testing.T) {
	fixed, msg := RunExpertMode("", "import * is not supported", 0, true)
	if !fixed || !strings.Contains(msg, "import *") || !strings.Contains(msg, "ROMA4D EXPERT DEBUG") {
		t.Fatalf("msg=%q", msg)
	}
}

func TestHandleFailure_noTTYNoHang(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	defer r.Close()
	// Non-interactive stdin (not a TTY); EOF so REPL exits immediately if entered.
	code := HandleFailure(FailureContext{
		Tool: "r4d", Verb: "build",
		SourcePath: "",
		RawError:   "some unknown compiler error",
		Stdin:      r,
		Stderr:     os.Stderr,
	})
	if code != 1 {
		t.Fatalf("exit=%d", code)
	}
}
