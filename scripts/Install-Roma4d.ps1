#Requires -Version 5.1
<#
.SYNOPSIS
  Universal Roma4D installer entrypoint — run from the 4DEngine repo root.

.DESCRIPTION
  Verifies roma4d/roma4d.toml exists, then runs roma4d\Install-Full.ps1 (Go check, mod download, tests, user PATH + R4D_PKG_ROOT).

.EXAMPLE
  cd C:\Users\Asus\Desktop\4DEngine
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Install-Roma4d.ps1
#>
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$roma = Join-Path $repoRoot "roma4d"
$toml = Join-Path $roma "roma4d.toml"

if (-not (Test-Path $toml)) {
    Write-Error @"
Expected Roma4D at:
  $toml

Run this script from a clone of 4DEngine (repo root must contain the folder 'roma4d').
"@
}

$full = Join-Path $roma "Install-Full.ps1"
if (-not (Test-Path $full)) {
    Write-Error "Missing $full"
}

if ($SkipTests) {
    & $full -SkipTests
} else {
    & $full
}
