#!/usr/bin/env sh
# Full Roma4D install for the 4DEngine monorepo (this directory: .../4DEngine/roma4d).
# - Checks go (1.22+) and clang
# - go mod download
# - go test ./...  (set SKIP_TESTS=1 to skip)
# - go install r4, r4d, roma4d with embedded package root
#
# After:  export PATH="$(go env GOPATH)/bin:$PATH"
#         export R4D_PKG_ROOT="$(pwd)"
# Optional: add both to your shell profile (~/.bashrc, ~/.zshrc).
set -e
cd "$(dirname "$0")"
ROOT="$(pwd -P 2>/dev/null || pwd)"
if ! test -f "$ROOT/roma4d.toml"; then
  echo "error: roma4d.toml not found — run from 4DEngine/roma4d" >&2
  exit 1
fi

echo "Roma4D full install — $ROOT"

if ! command -v go >/dev/null 2>&1; then
  echo "error: go not on PATH (install Go 1.22+ from https://go.dev/dl/)" >&2
  exit 1
fi

GOVER="$(go version)"
echo "Found: $GOVER"

if ! command -v clang >/dev/null 2>&1; then
  echo "warning: clang not on PATH — install llvm/clang (e.g. apt install clang, brew install llvm)" >&2
fi

echo ""
echo "go mod download"
go mod download

if test "${SKIP_TESTS:-0}" != "1"; then
  echo ""
  echo "go test ./..."
  go test ./...
fi

echo ""
ROOT_POSIX="$(printf '%s' "$ROOT" | sed 's|\\|/|g')"
go install -ldflags "-X github.com/RomanAILabs-Auth/Roma4D/internal/cli.EmbeddedPkgRoot=$ROOT_POSIX" ./cmd/r4 ./cmd/r4d ./cmd/roma4d

GOPATH="$(go env GOPATH)"
BIN="$GOPATH/bin"
echo ""
echo "Installed: $BIN/r4  $BIN/r4d  $BIN/roma4d"
echo ""
echo "Add to PATH and set R4D_PKG_ROOT (current session):"
echo "  export PATH=\"$BIN:\$PATH\""
echo "  export R4D_PKG_ROOT=\"$ROOT\""
echo ""
echo "Verify:"
echo "  r4d version"
echo "  r4d examples/min_main.r4d"
