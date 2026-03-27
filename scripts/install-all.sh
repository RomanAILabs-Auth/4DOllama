#!/usr/bin/env bash
# One-shot: 4DOllama (scripts/install.sh) then Roma4D (scripts/install-roma4d.sh).
# Run from repo root:
#   chmod +x scripts/install-all.sh && ./scripts/install-all.sh
#
# Environment:
#   SKIP_4DOLLAMA=1   — only install Roma4D
#   SKIP_ROMA4D=1     — only install 4DOllama
#   SKIP_TESTS=1      — passed through to Roma4d install (faster)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
test -f go.mod || { echo "error: go.mod not found — cd to repo root"; exit 1; }

echo ""
echo "========================================"
echo "  4DEngine — 4DOllama + Roma4D installer"
echo "========================================"
echo ""

if [[ "${SKIP_4DOLLAMA:-}" != "1" ]]; then
  bash "$ROOT/scripts/install.sh"
else
  echo "Skipped 4DOllama (SKIP_4DOLLAMA=1)."
fi

if [[ "${SKIP_ROMA4D:-}" != "1" ]]; then
  echo ""
  echo "=== Roma4D (r4d) ==="
  sh "$ROOT/scripts/install-roma4d.sh"
else
  echo "Skipped Roma4D (SKIP_ROMA4D=1)."
fi

echo ""
echo "Install-all finished. Open a new shell or: source ~/.profile"
echo "  4DOllama:  4dollama doctor && 4dollama run qwen2.5"
echo "  Roma4D:    cd roma4d && ./r4d.ps1 examples/min_main.r4d   # or see roma4d/README.md"
echo "  Guide:     docs/INSTALL_4DOLLAMA.md"
echo ""
