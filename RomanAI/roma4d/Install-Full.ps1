#Requires -Version 5.1
<#
.SYNOPSIS
  Full Roma4D install for the 4DEngine monorepo (this folder: ...\4DEngine\roma4d).

.DESCRIPTION
  1. Verifies Go 1.22+ on PATH.
  2. Warns if neither Zig nor clang is found (Windows needs one for linking native output).
  3. go mod download
  4. Optionally go test ./... (use -SkipTests to skip).
  5. Runs scripts\Install-R4dUserEnvironment.ps1 — builds r4 / r4d / roma4d with embedded R4D_PKG_ROOT,
     adds Go bin to your user PATH, sets R4D_PKG_ROOT and optional R4D_GNU_ROOT.

  After completion, open a NEW terminal and run:  r4d version
  Example:  r4d examples\min_main.r4d   (exit code 42 is intentional for min_main)

.PARAMETER SkipTests
  Do not run go test ./... before installing.

.EXAMPLE
  cd C:\Users\Asus\Desktop\4DEngine\roma4d
  .\Install-Full.ps1
#>
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RomaRoot = (Resolve-Path $PSScriptRoot).Path
$toml = Join-Path $RomaRoot "roma4d.toml"
if (-not (Test-Path $toml)) {
    Write-Error "roma4d.toml not found at $RomaRoot — run this script from the roma4d directory inside 4DEngine."
}

Write-Host "Roma4D full install — repo root: $RomaRoot" -ForegroundColor Cyan
Set-Location $RomaRoot

# --- Go ---
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "Go is not on PATH. Install Go 1.22+ from https://go.dev/dl/ and retry."
}
$goVerLine = (go version 2>&1) -join " "
Write-Host "Found: $goVerLine" -ForegroundColor DarkGray
if ($goVerLine -match "go version go1\.(\d+)") {
    $minor = [int]$Matches[1]
    if ($minor -lt 22) {
        Write-Error "Go 1.22+ required; got: $goVerLine"
    }
}

# --- Linker toolchain (Windows) ---
if ($IsWindows -or $env:OS -match "Windows") {
    $hasZig = [bool](Get-Command zig -ErrorAction SilentlyContinue)
    $hasClang = [bool](Get-Command clang -ErrorAction SilentlyContinue)
    if (-not $hasZig -and -not $hasClang) {
        Write-Warning "Neither 'zig' nor 'clang' found on PATH. Roma4D needs one of them to link native binaries on Windows."
        Write-Host "  Recommended: install Zig and add zig.exe to PATH — https://ziglang.org/download/" -ForegroundColor Yellow
        Write-Host "  Or: LLVM clang + MinGW (MSYS2); Install-R4dUserEnvironment sets R4D_GNU_ROOT when MSYS is detected." -ForegroundColor Yellow
    } elseif ($hasZig) {
        Write-Host "Linker: Zig available (default Windows path)." -ForegroundColor Green
    } else {
        Write-Host "Linker: clang available (Zig not found; Clang + MinGW path)." -ForegroundColor Green
    }
}

Write-Host "`ngo mod download" -ForegroundColor Cyan
go mod download

if (-not $SkipTests) {
    Write-Host "`ngo test ./..." -ForegroundColor Cyan
    go test ./...
}

$userInstaller = Join-Path $RomaRoot "scripts\Install-R4dUserEnvironment.ps1"
if (-not (Test-Path $userInstaller)) {
    Write-Error "Missing $userInstaller"
}

Write-Host "`nRunning user environment installer (build + PATH + R4D_PKG_ROOT)..." -ForegroundColor Cyan
& $userInstaller

Write-Host "`nDone. Open a new PowerShell window, then:" -ForegroundColor Green
Write-Host "  r4d version" -ForegroundColor White
Write-Host "  r4d examples\min_main.r4d" -ForegroundColor White
