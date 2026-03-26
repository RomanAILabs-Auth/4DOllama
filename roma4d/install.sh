#!/usr/bin/env sh
# One-line-style install: builds `r4d` and `roma4d` into $GOBIN or $GOPATH/bin.
cd "$(dirname "$0")" && go install ./cmd/r4d ./cmd/roma4d && echo "Installed r4d and roma4d — ensure your Go bin directory is on PATH."
