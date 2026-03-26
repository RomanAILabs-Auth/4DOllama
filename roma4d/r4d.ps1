# Roma4D: build r4 + r4d from THIS repo into GOPATH\bin, prepend PATH, then run (always fresh binary).
# Usage: .\r4d.ps1 run examples\min_main.r4s
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
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
