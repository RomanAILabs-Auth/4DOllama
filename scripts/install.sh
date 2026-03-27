#!/usr/bin/env bash
# One-click: Rust release + Go binary, CPU fallback, pull qwen2.5, background serve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
test -f go.mod || { echo "go.mod not found — run from repo root"; exit 1; }
command -v go >/dev/null || { echo "Install Go from https://go.dev/dl/"; exit 1; }

MANIFEST="$ROOT/4d-engine/Cargo.toml"
CARGO_OK=0
if command -v cargo >/dev/null 2>&1; then
  echo "cargo build --release --manifest-path $MANIFEST  (CPU + GPU sources)"
  if cargo build --release --manifest-path "$MANIFEST"; then
    CARGO_OK=1
  fi
else
  echo "cargo not found — skipping Rust (CGO_ENABLED=0)."
fi

export CGO_ENABLED=1
if [ "$CARGO_OK" -ne 1 ]; then export CGO_ENABLED=0; fi

echo "go build -o 4dollama ./cmd/4dollama (CGO_ENABLED=$CGO_ENABLED)"
if ! go build -o 4dollama ./cmd/4dollama; then
  export CGO_ENABLED=0
  echo "retry go build with CGO_ENABLED=0"
  go build -o 4dollama ./cmd/4dollama
fi

DEST="$HOME/.local/bin"
mkdir -p "$DEST"
install -m 0755 4dollama "$DEST/4dollama"
rm -f 4dollama

MODELS_DIR="${HOME}/.4dollama/models"
mkdir -p "$MODELS_DIR"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

# Force CPU when no CUDA and not macOS Metal (unless FOURD_GPU already set)
if [ -z "${FOURD_GPU:-}" ]; then
  if [ -d "/System/Library/Frameworks/Metal.framework" ]; then
    :
  elif [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] || [ -f /usr/lib/wsl/lib/libcuda.so ]; then
    :
  else
    export FOURD_GPU=cpu
    echo "FOURD_GPU=cpu (no Metal/CUDA driver path found — full 4D on CPU)."
  fi
fi

export PATH="$DEST:$PATH"
export FOURD_SHARE_OLLAMA="${FOURD_SHARE_OLLAMA:-true}"
export OLLAMA_MODELS="$OLLAMA_MODELS"
export FOURD_MODELS="$MODELS_DIR"
export FOURD_DEFAULT_MODEL="${FOURD_DEFAULT_MODEL:-qwen2.5}"
export FOURD_PORT="${FOURD_PORT:-13377}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export FOURD_INFERENCE="${FOURD_INFERENCE:-ollama}"

grep -q 'FOURD_MODELS' "$HOME/.profile" 2>/dev/null || echo "export FOURD_MODELS=\"$MODELS_DIR\"" >> "$HOME/.profile" || true
if [ "${FOURD_GPU:-}" = "cpu" ]; then
  grep -q 'export FOURD_GPU=cpu' "$HOME/.profile" 2>/dev/null || echo "export FOURD_GPU=cpu" >> "$HOME/.profile" || true
fi

case ":$PATH:" in *":$DEST:"*) ;; *)
  echo ""
  echo "Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
  ;;
esac

echo ""
echo "Pulling qwen2.5 (best-effort)…"
4dollama pull qwen2.5 || true

LOG_DIR="${HOME}/.4dollama"
mkdir -p "$LOG_DIR"
echo "Starting 4dollama serve in background (log: $LOG_DIR/serve.log)…"
nohup env FOURD_MODELS="$MODELS_DIR" OLLAMA_MODELS="$OLLAMA_MODELS" FOURD_PORT="${FOURD_PORT}" \
  FOURD_GPU="${FOURD_GPU:-}" OLLAMA_HOST="${OLLAMA_HOST}" \
  FOURD_INFERENCE="${FOURD_INFERENCE:-ollama}" 4dollama serve >>"$LOG_DIR/serve.log" 2>&1 &
disown 2>/dev/null || true

BASE="http://127.0.0.1:${FOURD_PORT}"
ok=0
for i in $(seq 1 80); do
  sleep 0.15
  if curl -sf "$BASE/healthz" >/dev/null; then ok=1; break; fi
done
if [ "$ok" -eq 1 ]; then
  echo "Serve is up on $BASE"
else
  echo "Warning: serve not responding yet — run: 4dollama serve"
fi

echo ""
echo "🎉 4DOllama is ready! Works on CPU or GPU. Just type: 4dollama run qwen2.5"
echo "   doctor: 4dollama doctor"
echo ""
4dollama version
