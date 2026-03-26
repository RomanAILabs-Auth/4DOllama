#!/usr/bin/env sh
# One-line-style install: builds `r4d` and `roma4d` into $GOBIN or $GOPATH/bin.
# Embeds this repo path so `r4d /anywhere/hello.r4d` works without cd into the clone.
set -e
cd "$(dirname "$0")"
ROOT="$(pwd -P 2>/dev/null || pwd)"
go install -ldflags "-X github.com/RomanAILabs-Auth/Roma4D/internal/cli.EmbeddedPkgRoot=$ROOT" ./cmd/r4 ./cmd/r4d ./cmd/roma4d
echo "Installed r4, r4d, and roma4d — ensure your Go bin directory is on PATH."
