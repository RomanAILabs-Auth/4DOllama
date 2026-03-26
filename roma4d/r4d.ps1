# r4d.ps1 - Build Roma4D tools, then run r4d (same as typing r4d after install)
# You do NOT need this file if you already ran scripts\Install-R4dUserEnvironment.ps1 once.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$gobin = ((go env GOBIN) -replace "`r|`n", "").Trim()
if ([string]::IsNullOrWhiteSpace($gobin)) {
    $gopath = ((go env GOPATH) -replace "`r|`n", "").Trim()
    $gobin = Join-Path $gopath "bin"
}

go build -o (Join-Path $gobin "r4.exe") ./cmd/r4
go build -o (Join-Path $gobin "r4d.exe") ./cmd/r4d
go build -o (Join-Path $gobin "roma4d.exe") ./cmd/roma4d

$env:Path = "$gobin;$env:Path"
$exe = Join-Path $gobin "r4d.exe"

if ($args.Count -eq 0) {
    Write-Host ""
    Write-Host "  ---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  HOW TO RUN A PROGRAM (type this in PowerShell):" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      r4d myfile.r4d" -ForegroundColor White
    Write-Host ""
    Write-Host "  File in another folder? Use the full path:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      r4d C:\Users\You\Desktop\myfile.r4d" -ForegroundColor White
    Write-Host ""
    Write-Host "  Folder name with spaces? Use quotes:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      r4d `"C:\My Stuff\hello.r4d`"" -ForegroundColor White
    Write-Host "  ---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    & $exe help
    exit $LASTEXITCODE
}

if (-not (Get-Command zig -ErrorAction SilentlyContinue)) {
    Write-Host "[Tip] Install Zig for the smoothest Windows builds: https://ziglang.org/download/" -ForegroundColor DarkYellow
}

& $exe @args
exit $LASTEXITCODE
