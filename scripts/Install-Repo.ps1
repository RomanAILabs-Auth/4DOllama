#Requires -Version 5.1
<#
.SYNOPSIS
  One-shot installer for this monorepo: 4DOllama first, then Roma4D (optional).

.DESCRIPTION
  Run from the repository root (the folder that contains go.mod and roma4d/).

  Step 1 — 4DOllama: builds Rust four_d_engine when possible, Go 4dollama CLI, copies to
  %USERPROFILE%\.4dollama\bin, sets user env vars, best-effort pull + background serve.

  Step 2 — Roma4D: runs scripts\Install-Roma4d.ps1 (Go r4d compiler, PATH, tests unless skipped).

.EXAMPLE
  # Double-click install.cmd, or:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1

.EXAMPLE
  # Only 4DOllama (skip Roma4D):
  .\scripts\Install-Repo.ps1 -SkipRoma4D

.EXAMPLE
  # Try to install Go via winget if missing:
  .\scripts\Install-Repo.ps1 -InstallGo

.EXAMPLE
  # Faster Roma4D (skip compiler tests):
  .\scripts\Install-Repo.ps1 -Roma4dSkipTests
#>
param(
    [switch]$Skip4DOllama,
    [switch]$SkipRoma4D,
    [switch]$InstallGo,
    [switch]$SkipCargo,
    [switch]$SkipBackup,
    [switch]$Roma4dSkipTests
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = Split-Path -Parent $here

Set-Location $root
if (-not (Test-Path (Join-Path $root "go.mod"))) {
    Write-Error "go.mod not found. Open PowerShell in the repo root (folder with go.mod), then run:`n  powershell -ExecutionPolicy Bypass -File .\scripts\Install-Repo.ps1"
}

function Write-Banner([string]$title) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

if (-not $Skip4DOllama) {
    Write-Banner "Step 1/2 — 4DOllama"
    $installFour = Join-Path $here "install.ps1"
    if (-not (Test-Path $installFour)) { Write-Error "Missing $installFour" }
    & $installFour -InstallGo:$InstallGo -SkipCargo:$SkipCargo -SkipBackup:$SkipBackup
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Skipped 4DOllama (-Skip4DOllama)." -ForegroundColor Yellow
}

if (-not $SkipRoma4D) {
    Write-Banner "Step 2/2 — Roma4D (r4d compiler)"
    $romaScript = Join-Path $here "Install-Roma4d.ps1"
    if (-not (Test-Path $romaScript)) {
        Write-Warning "Missing $romaScript — skipping Roma4D."
    } else {
        if ($Roma4dSkipTests) {
            & $romaScript -SkipTests
        } else {
            & $romaScript
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Roma4D install exited with code $LASTEXITCODE. 4DOllama may still be usable."
        }
    }
} else {
    Write-Host "Skipped Roma4D (-SkipRoma4D)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Install-Repo finished." -ForegroundColor Green
Write-Host "  Open a NEW terminal so PATH and environment variables apply." -ForegroundColor White
Write-Host "  4DOllama:  4dollama doctor   then   4dollama run qwen2.5" -ForegroundColor White
Write-Host "  Roma4D:    cd roma4d ; .\r4d.ps1 examples\min_main.r4d" -ForegroundColor White
Write-Host "  Full guide: docs\INSTALL_4DOLLAMA.md" -ForegroundColor Gray
Write-Host ""
