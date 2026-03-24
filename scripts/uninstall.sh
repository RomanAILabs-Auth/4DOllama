#!/usr/bin/env bash
# Remove ~/.local/bin/4dollama and strip FOURD_* lines from ~/.profile (best-effort).
# Usage: ./scripts/uninstall.sh   [--purge-models]   [--purge-data]
set -euo pipefail
DEST="${HOME}/.local/bin/4dollama"
PURGE_MODELS=0
PURGE_DATA=0
for a in "$@"; do
  case "$a" in
    --purge-models) PURGE_MODELS=1 ;;
    --purge-data) PURGE_DATA=1 ;;
  esac
done

echo "Stopping 4dollama serve (if running)…"
pkill -f '[4]dollama serve' 2>/dev/null || true
sleep 0.3

if [[ -f "$DEST" ]]; then
  rm -f "$DEST"
  echo "Removed $DEST"
fi

PROFILE="${HOME}/.profile"
if [[ -f "$PROFILE" ]]; then
  tmp="$(mktemp)"
  grep -v '^export FOURD_MODELS=' "$PROFILE" | grep -v '^export FOURD_GPU=cpu$' >"$tmp" || true
  mv "$tmp" "$PROFILE"
  echo "Stripped FOURD_MODELS / FOURD_GPU=cpu lines from ~/.profile (if present)."
fi

if [[ "$PURGE_MODELS" -eq 1 ]]; then
  rm -rf "${HOME}/.4dollama/models"
  echo "Removed ~/.4dollama/models"
fi
if [[ "$PURGE_DATA" -eq 1 ]]; then
  rm -rf "${HOME}/.4dollama"
  echo "Removed ~/.4dollama"
fi

echo ""
echo "4DOllama CLI removed from ~/.local/bin. Re-login or fix PATH; reinstall: ./scripts/install.sh"
