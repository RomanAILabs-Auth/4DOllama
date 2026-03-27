# Launch-4DOllama.ps1 — build and run native 4D substrate (fourd) from monorepo root.
#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "go.mod"))) {
    Write-Error "Run from 4DEngine tree; go.mod not found at $Root"
}
Set-Location $Root
$exe = Join-Path $Root "4dollama.exe"
go build -o $exe ./cmd/4dollama
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $exe fourd ga-demo
& $exe fourd lattice -steps 60 @args
