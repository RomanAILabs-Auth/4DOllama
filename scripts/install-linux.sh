#!/usr/bin/env bash
set -euo pipefail
# One-shot build from source (Debian/Ubuntu-style hosts with Docker).
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t fourdollama:latest "$ROOT"
echo "Built fourdollama:latest — run with:"
echo "  docker run --rm -p 13373:13373 -v fourd_models:/models fourdollama:latest"
