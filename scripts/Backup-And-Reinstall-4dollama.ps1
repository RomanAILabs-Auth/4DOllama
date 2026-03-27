#Requires -Version 5.1
<#
.SYNOPSIS
  Backs up the current 4dollama.exe then rebuilds and copies the latest from this repo.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Backup-And-Reinstall-4dollama.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$installDir = Join-Path $env:USERPROFILE ".4dollama\bin"
$exe = Join-Path $installDir "4dollama.exe"
$backupRoot = Join-Path $env:USERPROFILE ".4dollama\backup"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null

if (Test-Path $exe) {
    $bak = Join-Path $backupRoot "4dollama-$stamp.exe"
    Copy-Item -LiteralPath $exe -Destination $bak -Force
    Write-Host "Backed up to: $bak" -ForegroundColor Green
} else {
    Write-Host "No existing $exe to back up." -ForegroundColor Yellow
}

Get-CimInstance Win32_Process -Filter "Name = '4dollama.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 800

$go = Get-Command go -ErrorAction SilentlyContinue
if (-not $go) {
    Write-Error "Go not on PATH. Install Go or run full Install-Repo.ps1 -InstallGo"
}

$out = Join-Path $root "4dollama.exe"
& go build -o $out .\cmd\4dollama
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Copy-Item -Force $out $exe
Unblock-File -Path $exe -ErrorAction SilentlyContinue
Write-Host "Installed: $exe" -ForegroundColor Green
& $exe version
