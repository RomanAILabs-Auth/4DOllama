# Roma4D: build r4 + r4d from THIS repo into GOPATH\bin, prepend PATH, then run (always fresh binary).
# Windows native link: Zig (`zig cc`) when `zig` is on PATH; else LLVM clang + MinGW.
# Usage: .\r4d.ps1 examples\min_main.r4d
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command zig -ErrorAction SilentlyContinue)) {
    Write-Host "[r4d.ps1] Zig not on PATH — install from https://ziglang.org/download/ for the default Windows toolchain (or use clang+MinGW)." -ForegroundColor DarkYellow
}
$gobin = Join-Path (go env GOPATH) "bin"
New-Item -ItemType Directory -Force -Path $gobin | Out-Null
go build -o (Join-Path $gobin "r4.exe") ./cmd/r4
go build -o (Join-Path $gobin "r4d.exe") ./cmd/r4d
go build -o (Join-Path $gobin "roma4d.exe") ./cmd/roma4d
$env:Path = "$gobin;$env:Path"
$exe = Join-Path $gobin "r4.exe"
Write-Host ">> $exe" -ForegroundColor DarkGray
& $exe @args
exit $LASTEXITCODE
