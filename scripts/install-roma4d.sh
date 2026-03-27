#!/usr/bin/env sh
# Universal Roma4d installer entrypoint — run from the 4DEngine repo root.
# Usage:
#   chmod +x scripts/install-roma4d.sh roma4d/install-full.sh
#   ./scripts/install-roma4d.sh
# Skip tests: SKIP_TESTS=1 ./scripts/install-roma4d.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROMA="$ROOT/roma4d"
if ! test -f "$ROMA/roma4d.toml"; then
  echo "error: expected Roma4D at $ROMA/roma4d.toml" >&2
  echo "  Run this script from the 4DEngine repo root (folder must contain roma4d/)." >&2
  exit 1
fi
exec sh "$ROMA/install-full.sh"
