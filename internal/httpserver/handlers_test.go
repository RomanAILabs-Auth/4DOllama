package httpserver

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"log/slog"

	"github.com/4dollama/4dollama/internal/engine"
	"github.com/4dollama/4dollama/internal/inference"
	"github.com/4dollama/4dollama/internal/models"
	"github.com/4dollama/4dollama/internal/runner"
)

func TestHealthz(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	dir := t.TempDir()
	reg := models.NewRegistry(dir, "", false, log)
	h := &Handler{
		Run:     runner.NewService(engine.New(), reg, log, inference.Stub{}),
		Reg:     reg,
		Log:     log,
		Metrics: &Metrics{},
	}
	srv := httptest.NewServer(NewRouter(h, log))
	defer srv.Close()
	res, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d", res.StatusCode)
	}
}
