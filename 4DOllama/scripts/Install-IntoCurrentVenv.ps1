# Install-IntoCurrentVenv.ps1
# Copyright RomanAILabs - Daniel Harding
# Christ is King.
#Requires -Version 5.1
# Run AFTER activating quantum_win (or any venv): installs 4dollam/4dollama into that env's Scripts.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
python -m pip install -e .
$prefix = (python -c "import sys; print(sys.prefix)" 2>$null).Trim()
if (-not $prefix) {
    Write-Error "python not on PATH — activate quantum_win first."
}
$scripts = Join-Path $prefix "Scripts"
Write-Host ""
Write-Host "Installed fourdollama. Console apps:" -ForegroundColor Green
Write-Host "  $scripts\4dollam.exe"
Write-Host "  $scripts\4dollama.exe"
Write-Host ""
if (-not ($env:Path -split ';' | Where-Object { $_ -eq $scripts })) {
    Write-Host "This session does not have Scripts on PATH. For THIS window only, run:" -ForegroundColor Yellow
    Write-Host "  `$env:Path = `"$scripts;`$env:Path`""
    Write-Host ""
    Write-Host "Then:  4dollam run qwen2.5" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or always use (no PATH needed):" -ForegroundColor Yellow
    Write-Host "  python -m fourdollama run qwen2.5"
} else {
    Write-Host "Try:  4dollam run qwen2.5" -ForegroundColor Cyan
}
